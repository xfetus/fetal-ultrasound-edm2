# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import pandas as pd
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        use_labels  = True,     # Enable conditioning labels? False = label dimension is zero.
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self): # [CHW]
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = anything goes.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        ext = self._file_ext(fname)
        with self._open_file(fname) as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
            else:
                image = np.array(PIL.Image.open(f))
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
# Dataset class designed specifically for the Ultrasound dataset

class UltrasoundDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = anything goes.
        fpus23_path     = None, # Path to FPUS23 dataset (optional).
        african_path    = None, # Path to African Zenodo dataset root (optional).
        fetal_abdomen_path = None, # Path to fetal abdominal segmentation IMAGES dir (optional).
        split           = 'train', # 'train' or 'val'
        val_phantom_size   = 30,   # FPUS23 images reserved for validation
        num_classes     = None, # Override label dimension (e.g. for val set to match training).
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self._type = 'dir'
        self._all_fnames = None

        # Define classes and give each int labels
        self.plane_classes = {
            'Other' : 0,
            'Maternal cervix' : 1,
            'Fetal abdomen' : 2,
            'Fetal brain' : 3,
            'Fetal femur' : 4,
            'Fetal thorax' : 5,
        }

        self.img_dir = os.path.join(self._path, "Images")
        self.csv_file = os.path.join(self._path, "FETAL_PLANES_DB_data.csv")

        # Load the csv as a pandas data frame
        self.meta_data = pd.read_csv(self.csv_file, delimiter=';')

        # Filter DataFrame to train or test split
        self.meta_data = self.meta_data[self.meta_data['Train '] == (1 if split == 'train' else 0)]
        
        self.meta_data["Image_name"] = self.meta_data["Image_name"].astype(str) + ".png"

        # Build combined image paths and labels from Spanish dataset
        self._image_paths = []
        self._labels = []
        for _, row in self.meta_data.iterrows():
            img_path = os.path.join(self.img_dir, row["Image_name"])
            self._image_paths.append(img_path)
            self._labels.append(self.plane_classes[row['Plane']])

        # Add FPUS23 dataset images if provided
        if fpus23_path is not None:
            fpus23_label = 6
            fpus23_paths = []
            # Collect from Dataset_Plane (class-organized images)
            plane_dir = os.path.join(fpus23_path, 'Dataset_Plane')
            if os.path.isdir(plane_dir):
                for class_name in sorted(os.listdir(plane_dir)):
                    class_dir = os.path.join(plane_dir, class_name)
                    if os.path.isdir(class_dir):
                        for fname in sorted(os.listdir(class_dir)):
                            if self._file_ext(fname) in {'.png', '.jpg', '.jpeg'}:
                                fpus23_paths.append(os.path.join(class_dir, fname))
            # Collect from Dataset/four_poses (stream-organized frames)
            poses_dir = os.path.join(fpus23_path, 'Dataset', 'four_poses')
            if os.path.isdir(poses_dir):
                for stream_name in sorted(os.listdir(poses_dir)):
                    stream_dir = os.path.join(poses_dir, stream_name)
                    if os.path.isdir(stream_dir):
                        for fname in sorted(os.listdir(stream_dir)):
                            if self._file_ext(fname) in {'.png', '.jpg', '.jpeg'}:
                                fpus23_paths.append(os.path.join(stream_dir, fname))
            # Deterministic train/val split: last val_phantom_size go to val
            fpus23_paths = sorted(fpus23_paths)
            if split == 'val':
                fpus23_paths = fpus23_paths[-val_phantom_size:] if val_phantom_size > 0 else []
            else:
                fpus23_paths = fpus23_paths[:-val_phantom_size] if val_phantom_size > 0 else fpus23_paths
            for p in fpus23_paths:
                self._image_paths.append(p)
                self._labels.append(fpus23_label)

        # Add African dataset images if provided (each country is a subdirectory).
        if african_path is not None:
            african_label = 7
            for country_name in sorted(os.listdir(african_path)):
                country_dir = os.path.join(african_path, country_name)
                if os.path.isdir(country_dir):
                    for fname in sorted(os.listdir(country_dir)):
                        if self._file_ext(fname) in {'.png', '.jpg', '.jpeg'}:
                            self._image_paths.append(os.path.join(country_dir, fname))
                            self._labels.append(african_label)

        # Add fetal abdominal segmentation images if provided (flat directory).
        if fetal_abdomen_path is not None:
            fetal_label = 8
            for fname in sorted(os.listdir(fetal_abdomen_path)):
                if self._file_ext(fname) in {'.png', '.jpg', '.jpeg'}:
                    self._image_paths.append(os.path.join(fetal_abdomen_path, fname))
                    self._labels.append(fetal_label)

        self._labels = np.array(self._labels, dtype=np.int64)
        self._all_fnames = [os.path.basename(p) for p in self._image_paths]

        PIL.Image.init()
        supported_ext = PIL.Image.EXTENSION.keys() | {'.npy'}
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in supported_ext)
        if len(self._image_paths) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_paths), 8, 64, 64]
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
        if num_classes is not None:
            self._label_shape = [num_classes]

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        flip = idx >= len(self._raw_idx)  # Second half of indices are flipped
        cache_key = (raw_idx, flip)
        
        image = self._cached_images.get(cache_key, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if flip:
                image = np.flip(image, axis=2)  # Horizontal flip on width axis (C, H, W format)
            with torch.no_grad():
                image = self.encoder.encode_pixels(torch.as_tensor(image).to(self.device).unsqueeze(0))
                image = image[0].cpu().numpy()
            if self._cache:
                self._cached_images[cache_key] = image
        assert isinstance(image, np.ndarray)
        return image.copy(), self.get_label(raw_idx)

    def _load_raw_image(self, raw_idx):
        fname = self._image_paths[raw_idx]
        ext = self._file_ext(fname)
        with open(fname, 'rb') as f:
            if ext == '.npy':
                image = np.load(f)
                image = image.reshape(-1, *image.shape[-2:])
            elif ext == '.png' and pyspng is not None:
                image = pyspng.load(f.read())
                h, w = image.shape[:2]
                size = min(h, w)
                top = (h - size) // 2
                left = (w - size) // 2
                image = image[top:top+size, left:left+size]
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
                image = np.mean(image, axis=0)
                image = np.stack([image, image, image])
            else:
                image = np.array(PIL.Image.open(f))
                h, w = image.shape[:2]
                size = min(h, w)
                top = (h - size) // 2
                left = (w - size) // 2
                image = image[top:top+size, left:left+size]
                image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_LINEAR)
                image = image.reshape(*image.shape[:2], -1).transpose(2, 0, 1)
                image = np.mean(image, axis=0)
                image = np.stack([image, image, image])
        
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels
    
    def _get_raw_labels(self):
        return self._labels
#----------------------------------------------------------------------------