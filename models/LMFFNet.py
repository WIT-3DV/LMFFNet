import torch
import torch.nn as nn
import torch.nn.functional as nnf
import utils

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, batch_norm=True, preactivation=False):
        super().__init__()

        constant_pad = torch.nn.ConstantPad3d
        conv = torch.nn.Conv3d
        bn = torch.nn.BatchNorm3d

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(tuple([padding % 2, padding - padding % 2] * 3), 0)
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [torch.nn.ReLU(), pad, conv(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride)]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [pad, conv(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride)]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class LFFBlock(nn.Module):

    def __init__(self, in_channel, kernel_size, unit, growth_rate):
        super(LFFBlock, self).__init__()

        self.conv_units = torch.nn.ModuleList()
        for i in range(unit):
            self.conv_units.append(
                ConvBlock(
                    in_channel=in_channel,
                    out_channel=growth_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    batch_norm=True,
                    preactivation=True
                )
            )
            in_channel += growth_rate

    def forward(self, x):
        stack_feature = None

        for i, conv in enumerate(self.conv_units):
            if stack_feature is None:
                inputs = x
            else:
                inputs = torch.cat([x, stack_feature], dim=1)
            out = conv(inputs)
            if stack_feature is None:
                stack_feature = out
            else:
                stack_feature = torch.cat([stack_feature, out], dim=1)

        return torch.cat([x, stack_feature], dim=1)


class DownSampleBlock(nn.Module):

    def __init__(self, in_channel, base_channel, kernel_size, unit, growth_rate, skip_channel=None, downsample=True, skip=True):
        super(DownSampleBlock, self).__init__()
        self.skip = skip

        self.downsample = ConvBlock(in_channel=in_channel, out_channel=base_channel, kernel_size=kernel_size, stride=(2 if downsample else 1), batch_norm=True, preactivation=True)

        self.lff = LFFBlock(in_channel=base_channel, kernel_size=3, unit=unit, growth_rate=growth_rate)

        if skip:
            self.skip_conv = ConvBlock(in_channel=base_channel + unit * growth_rate, out_channel=skip_channel, kernel_size=3, stride=1, batch_norm=True, preactivation=True)

    def forward(self, x):
        x = self.downsample(x)
        x = self.lff(x)

        if self.skip:
            x_skip = self.skip_conv(x)
            return x, x_skip
        else:
            return x

class DWCBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, batch_norm=True, preactivation=False):
        super(DWCBlock, self).__init__()

        constant_pad = torch.nn.ConstantPad3d
        conv = torch.nn.Conv3d
        bn = torch.nn.BatchNorm3d

        padding = kernel_size - stride
        if padding % 2 != 0:
            pad = constant_pad(
                tuple([padding % 2, padding - padding % 2] * 3), 0
            )
        else:
            pad = constant_pad(padding // 2, 0)

        if preactivation:
            layers = [torch.nn.ReLU(), pad, conv( in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride, groups=in_channel, bias=False),
                conv(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=True)]
            if batch_norm:
                layers = [bn(in_channel)] + layers
        else:
            layers = [pad, conv( in_channels=in_channel, out_channels=in_channel, kernel_size=kernel_size, stride=stride, groups=in_channel, bias=False),
                      conv(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)]
            if batch_norm:
                layers.append(bn(out_channel))
            layers.append(torch.nn.ReLU())

        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class LMFFNet(nn.Module):
    def __init__(self, config):
        super(LMFFNet, self).__init__()
        self.scaling_version = config.scaling_version
        self.img_size = config.img_size
        base_channels = config.base_channels
        skip_channels = config.skip_channels

        units = config.units
        pmfs_ch = config.pmfs_ch
        upsample_mode = 'trilinear'
        kernel_sizes = [5, 3, 3]
        growth_rates = [4, 8, 16]
        downsample_channels = [base_channels[i] + units[i] * growth_rates[i] for i in range(len(base_channels))] # [44, 128, 224]

        # 构建PMFSNet中的下采样模块
        self.down_convs = nn.ModuleList()
        for i in range(3):
            self.down_convs.append(
                DownSampleBlock(
                    in_channel=(2 if i == 0 else downsample_channels[i - 1]),
                    base_channel=base_channels[i],
                    kernel_size=kernel_sizes[i],
                    skip_channel=skip_channels[i],
                    unit=units[i],
                    growth_rate=growth_rates[i],
                    downsample=True,
                    skip=((i < 2) if self.scaling_version == "BASIC" else True),
                )
            )

        # 根据缩放版本构建上采样和解码器部分
        if self.scaling_version == "BASIC":
            self.up2 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv2 = DownSampleBlock(in_channel=downsample_channels[2] + skip_channels[1],
                                                         base_channel=base_channels[1],
                                                         kernel_size=3,
                                                         unit=units[1],
                                                         growth_rate=growth_rates[1],
                                                         downsample=False,
                                                         skip=False)
            self.up1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.up_conv1 = DownSampleBlock(in_channel=downsample_channels[1] + skip_channels[0],
                                                         base_channel=base_channels[0],
                                                         kernel_size=3,
                                                         unit=units[0],
                                                         growth_rate=growth_rates[0],
                                                         downsample=False,
                                                         skip=False)
        else:
            self.bottle_conv = ConvBlock(
                in_channel=downsample_channels[2] + skip_channels[2],
                out_channel=skip_channels[2],
                kernel_size=3,
                stride=1,
                batch_norm=True,
                preactivation=True
            )
            self.upsample_1 = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
            self.upsample_2 = torch.nn.Upsample(scale_factor=4, mode=upsample_mode)

        # 输出卷积层
        self.out_conv = ConvBlock(
            in_channel=(downsample_channels[0] if self.scaling_version == "BASIC" else sum(skip_channels)),
            out_channel=3,  # 假设输出通道为3，根据实际调整
            kernel_size=3,
            stride=1,
            batch_norm=True,
            preactivation=True
        )
        self.upsample_out = torch.nn.Upsample(scale_factor=2, mode=upsample_mode)
        self.spatial_trans = utils.register_model(config.img_size)

    def forward(self, x):
        source = x[:, 0:1, :, :]

        if self.scaling_version == "BASIC":
            x1, x1_skip = self.down_convs[0](x)
            x2, x2_skip = self.down_convs[1](x1)
            x3 = self.down_convs[2](x2)
            d3 = self.Global([x1, x2, x3])
            d2 = self.up2(d3)
            d2 = torch.cat((x2_skip, d2), dim=1)
            d2 = self.up_conv2(d2)
            d1 = self.up1(d2)
            d1 = torch.cat((x1_skip, d1), dim=1)
            d1 = self.up_conv1(d1)
            out = self.out_conv(d1)
        else:
            x1, skip1 = self.down_convs[0](x)
            x2, skip2 = self.down_convs[1](x1)
            x3, skip3 = self.down_convs[2](x2)
            x3 = self.Global([x1, x2, x3])
            skip3 = self.bottle_conv(torch.cat([x3, skip3], dim=1))
            skip2 = self.upsample_1(skip2)
            skip3 = self.upsample_2(skip3)
            out = self.out_conv(torch.cat([skip1, skip2, skip3], dim=1))

        flow = nnf.interpolate(out, size=self.img_size, mode='trilinear', align_corners=True)
        out = self.spatial_trans(source, flow)

        return out, flow

