import itertools
import os, glob
import random

import torch, sys
from torch.utils.data import Dataset
from .data_utils import pkload
import SimpleITK as sitk
import matplotlib.pyplot as plt
from torchvision import transforms
import collections



import numpy as np
class Base(object):
    def sample(self, *shape):
        return shape

    def tf(self, img, k=0):
        return img

    def __call__(self, img, dim=3, reuse=False): # class -> func()
        # image: nhwtc
        # shape: no first dim
        if not reuse:
            im = img if isinstance(img, np.ndarray) else img[0]
            # how to know  if the last dim is channel??
            # nhwtc vs nhwt??
            shape = im.shape[1:dim+1]
            # print(dim,shape) # 3, (240,240,155)
            self.sample(*shape)

        if isinstance(img, collections.Sequence):
            return [self.tf(x, k) for k, x in enumerate(img)] # img:k=0,label:k=1

        return self.tf(img)

    def __str__(self):
        return 'Identity()'

Identity = Base
class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ', '.join([str(s) for s in self.types])
        return 'NumpyType(({}))'.format(s)

class OASIS_BrainDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        self.transforms = transforms.Compose([NumpyType((np.float32, np.float16)),])

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        tar_list = self.paths.copy()
        tar_list.remove(path)
        random.shuffle(tar_list)
        tar_file = tar_list[0]
        x, x_seg = pkload(path)
        y, y_seg = pkload(tar_file)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)  # [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)


class OASIS_BrainInferDataset(Dataset):
    def __init__(self, data_path):
        self.paths = data_path
        self.transforms = transforms.Compose([NumpyType((np.float32, np.int16)),])

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, y, x_seg, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg = x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)



def load_validation_pair(data_path, filename1, filename2):
    # Load images and labels
    nim1 = os.path.join(data_path, filename1, 'slice_norm.nii.gz')
    #image1 = nim1.get_data()[:, :, 0]
    image1 = sitk.GetArrayFromImage(sitk.ReadImage(nim1))
    #[np.newaxis, ...]

    nim2 = os.path.join(data_path, filename2, 'slice_norm.nii.gz')
    image2 = sitk.GetArrayFromImage(sitk.ReadImage(nim2))#[np.newaxis, ...]
    # image2 = np.array(image2, dtype='float32')

    nim5 = os.path.join(data_path, filename1, 'slice_seg24.nii.gz')
    image5 = sitk.GetArrayFromImage(sitk.ReadImage(nim5))#[np.newaxis, ...]
    #image5 = np.array(image5, dtype='float32')
    # image5 = image5 / 35.0
    nim6 = os.path.join(data_path, filename2, 'slice_seg24.nii.gz')
    image6 = sitk.GetArrayFromImage(sitk.ReadImage(nim6))#[np.newaxis, ...]
    #image6 = np.array(image6, dtype='float32')  # 0 - 35 -》 0- 1
    # image6 = image6 / 35.0

    #image1 = np.reshape(image1, (1,) + image1.shape)
    #image2 = np.reshape(image2, (1,) + image2.shape)
    #image5 = np.reshape(image5, (1,) + image5.shape)
    #image6 = np.reshape(image6, (1,) + image6.shape)
    return image1, image2, image5, image6

def load_train_pair(data_path, filename1, filename2):
    # Load images and labels
    nim1 = os.path.join(data_path, filename1, 'slice_norm.nii.gz')
    image1 = sitk.GetArrayFromImage(sitk.ReadImage(nim1))#[np.newaxis, ...]
    #image1 = np.array(image1, dtype='float32')

    nim2 = os.path.join(data_path, filename2, 'slice_norm.nii.gz')
    image2 = sitk.GetArrayFromImage(sitk.ReadImage(nim2))#[np.newaxis, ...]
    #image2 = np.array(image2, dtype='float32')
    return image1 , image2

class TrainDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_path, img_file=None, trainingset=1):
        'Initialization'
        super(TrainDataset, self).__init__()
        self.data_path = data_path
        self.names = np.loadtxt(os.path.join(self.data_path, img_file), dtype='str')
        if trainingset == 1:
            self.filename = list(zip(self.names[:-1], self.names[1:]))
            assert len(self.filename) == 200, "Oh no! # of images != 200."
        elif trainingset == 2:
            self.filename = list(zip(self.names[1:], self.names[:-1]))
            assert len(self.filename) == 200, "Oh no! # of images != 200."
        elif trainingset == 3:
            self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
            self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
            self.filename = self.zip_filename_1 + self.zip_filename_2
            assert len(self.filename) == 400, "Oh no! # of images != 400."
        elif trainingset == 4:
            self.filename = list(itertools.permutations(self.names, 2))
            # print(len(self.names))
            # print(len(self.filename))
            assert len(self.filename) == 40200, "Oh no! # of images != 40200."

        else:
            assert 0 == 1, print('TrainDataset Invalid!')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # print('Total # of Images:   ', len(self.filename))
        # 154842
        # print(self.filename)
        mov_img, fix_img = load_train_pair(self.data_path, self.filename[index][0], self.filename[index][1])
        return mov_img, fix_img  # , mov_lab, fix_lab


class ValidationDataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data_path, img_file=None):
        'Initialization'
        super(ValidationDataset, self).__init__()
        # self.data_path = data_path
        # self.filename = pd.read_csv(os.path.join(data_path,'pairs_val.csv')).values
        # #print(self.filename)
        self.data_path = data_path
        self.names = np.loadtxt(os.path.join(self.data_path, img_file), dtype='str')
        self.zip_filename_1 = list(zip(self.names[:-1], self.names[1:]))
        self.zip_filename_2 = list(zip(self.names[1:], self.names[:-1]))
        self.filename = self.zip_filename_1 + self.zip_filename_2

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.filename)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        img_A, img_B, label_A, label_B = load_validation_pair(self.data_path, self.filename[index][0],
                                                              self.filename[index][1])
        # return self.filename[index][0], self.filename[index][1], img_A, img_B, label_A, label_B
        return img_A, img_B, label_A, label_B

class Scan_Dataset_1(Dataset):
    def __init__(self, dataset, file_dir, root, label_dir):
        # initialization
        self.dataset_file = os.path.join(root, "Dataset", dataset)
        self.files = sorted(glob.glob(os.path.join(self.dataset_file, file_dir, '*.nii.gz')))
        self.root = root
        self.label_dir = label_dir
        self.index_pair = list(itertools.permutations(self.files, 2))

    def group_label(self, label):
        GROUP_CONFIG = [
            ("frontal_lobe", 21, 34),
            ("parietal_lobe", 41, 50),
            ("occipital_lobe", 61, 68),
            ("temporal_lobe", 81, 92),
            ("cingulate_lobe", 101, 122),
            ("putamen", 163, 166),
            ("hippocampus", 181, 182)
        ]

        label_merged = np.zeros(label.shape, dtype=np.int32)
        for i, (name, start, end) in enumerate(GROUP_CONFIG):
            region = np.logical_and(label >= start, label <= end)
            label_merged[region] = i + 1
        return label_merged

    def __len__(self):
        # Returns the size of the dataset
        return len(self.index_pair)-1

    def __getitem__(self, index):
        # Index a certain data in the data set, you can also preprocess the data
        fixed_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.index_pair[index][0]))[np.newaxis, ...]
        moving_img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.index_pair[index][1]))[np.newaxis, ...]

        fixed_name = os.path.split(self.index_pair[index][0])[1]
        moving_name = os.path.split(self.index_pair[index][1])[1]
        fixed_label_file = glob.glob(os.path.join(self.dataset_file, self.label_dir, fixed_name[:4] + "*"))[0]
        fixed_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label_file))[np.newaxis, ...]
        moving_label_file = glob.glob(os.path.join(self.dataset_file, self.label_dir, moving_name[:4] + "*"))[0]
        moving_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_label_file))[np.newaxis, ...]

        # The return value is automatically converted to torch's tensor type
        return fixed_img_arr, moving_img_arr, self.group_label(fixed_label), self.group_label(moving_label)

class IXIBrainDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.paths = data_path
        self.atlas_path = atlas_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        #print(x.shape)
        #print(x.shape)
        #print(np.unique(y))
        # print(x.shape, y.shape)#(240, 240, 155) (240, 240, 155)
        # transforms work with nhwtc
        x, y = x[None, ...], y[None, ...]
        # print(x.shape, y.shape)#(1, 240, 240, 155) (1, 240, 240, 155)
        x,y = self.transforms([x, y])
        #y = self.one_hot(y, 2)
        #print(y.shape)
        #sys.exit(0)
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        #plt.figure()
        #plt.subplot(1, 2, 1)
        #plt.imshow(x[0, :, :, 8], cmap='gray')
        #plt.subplot(1, 2, 2)
        #plt.imshow(y[0, :, :, 8], cmap='gray')
        #plt.show()
        #sys.exit(0)
        #y = np.squeeze(y, axis=0)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x, y

    def __len__(self):
        return len(self.paths)


class IXIBrainInferDataset(Dataset):
    def __init__(self, data_path, atlas_path, transforms):
        self.atlas_path = atlas_path
        self.paths = data_path
        self.transforms = transforms

    def one_hot(self, img, C):
        out = np.zeros((C, img.shape[1], img.shape[2], img.shape[3]))
        for i in range(C):
            out[i,...] = img == i
        return out

    def __getitem__(self, index):
        path = self.paths[index]
        x, x_seg = pkload(self.atlas_path)
        y, y_seg = pkload(path)
        x, y = x[None, ...], y[None, ...]
        x_seg, y_seg= x_seg[None, ...], y_seg[None, ...]
        x, x_seg = self.transforms([x, x_seg])
        y, y_seg = self.transforms([y, y_seg])
        x = np.ascontiguousarray(x)# [Bsize,channelsHeight,,Width,Depth]
        y = np.ascontiguousarray(y)
        x_seg = np.ascontiguousarray(x_seg)  # [Bsize,channelsHeight,,Width,Depth]
        y_seg = np.ascontiguousarray(y_seg)
        x, y, x_seg, y_seg = torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(x_seg), torch.from_numpy(y_seg)
        return x, y, x_seg, y_seg

    def __len__(self):
        return len(self.paths)