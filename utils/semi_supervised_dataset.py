# -*- coding: utf-8 -*-
""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from skimage import io
import scipy.io as sio
import cv2
from glob import glob


class SemiSupervisedCountingDataset(Dataset):
    ""

    def __init__(self, root, labeled_list_path=None, DIV=32, crop_size=512,
                 is_training=True, labeled_ratio=None):
        ""
        self.root = root
        self.im_dir = os.path.join(root, 'images')
        self.gt_dir = os.path.join(root, 'gtdens')
        self.DIV = DIV
        self.crop_size = crop_size
        self.is_training = is_training


        self.rgb_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.rgb_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


        self.filelist = sorted(glob(os.path.join(self.im_dir, '*.jpg')))
        if len(self.filelist) == 0:

            self.filelist = sorted(glob(os.path.join(self.im_dir, '*.png')))

        if len(self.filelist) == 0:
            raise ValueError(f"No images found in {self.im_dir}")


        self._setup_labels(labeled_list_path, labeled_ratio)

        print(f"SemiSupervisedDataset: {len(self.filelist)} total, "
              f"{len(self.labeled_idx)} labeled, {len(self.unlabeled_idx)} unlabeled")

    def _setup_labels(self, labeled_list_path, labeled_ratio):
        ""
        self.labels = np.ones(len(self.filelist), dtype=bool)

        if labeled_list_path and os.path.exists(labeled_list_path):

            labeled_files = set()
            with open(labeled_list_path, 'r') as f:
                for line in f:
                    labeled_files.add(line.strip())


            self.labels = np.array([
                os.path.basename(f) in labeled_files
                for f in self.filelist
            ])

        elif labeled_ratio is not None and labeled_ratio < 1.0:

            n_labeled = int(len(self.filelist) * labeled_ratio)
            indices = np.random.permutation(len(self.filelist))
            self.labels = np.zeros(len(self.filelist), dtype=bool)
            self.labels[indices[:n_labeled]] = True

        self.labeled_idx = np.where(self.labels)[0]
        self.unlabeled_idx = np.where(~self.labels)[0]

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        try:

            img_path = self.filelist[idx]
            image = io.imread(img_path)


            if len(image.shape) < 3:
                image = np.expand_dims(image, axis=2)
            if image.shape[2] == 1:
                image = np.concatenate((image, image, image), axis=2)


            filename = os.path.basename(img_path)
            name, _ = os.path.splitext(filename)


            mat_path = os.path.join(self.gt_dir, f'{name}.mat')
            try:
                mat = sio.loadmat(mat_path)
                all_num = mat['dot_map'].sum().reshape((1, 1))
                density_map = mat['density_map'].astype(np.float32)
                seg_map = mat['seg_map'].astype(np.float32)
            except Exception as e:
                print(f"Warning: Error loading {mat_path}: {e}")
                H, W = image.shape[:2]
                all_num = np.array([[0.0]])
                density_map = np.zeros((H, W), dtype=np.float32)
                seg_map = np.zeros((H, W), dtype=np.float32)


            if self.is_training:
                image, density_map, seg_map = self._augment(image, density_map, seg_map)


            image_raw = image.astype(np.float32) / 255.0
            image_raw = torch.from_numpy(image_raw.transpose((2, 0, 1))).float()


            image = image.astype(np.float32) / 255.0 - self.rgb_mean
            image = image / self.rgb_std
            image = torch.from_numpy(image.transpose((2, 0, 1))).float()


            density_map = torch.from_numpy(density_map).unsqueeze(0).float()
            seg_map = torch.from_numpy(seg_map).unsqueeze(0).float()
            all_num = torch.from_numpy(all_num.astype(np.float32))


            image = self._pad_tensor(image)
            image_raw = self._pad_tensor(image_raw)
            density_map = self._pad_tensor(density_map)
            seg_map = self._pad_tensor(seg_map)


            weak_image = image.clone()
            strong_image = self._strong_augment(image.clone())

            is_labeled = self.labels[idx]

            return {
                'weak_image': weak_image,
                'strong_image': strong_image,
                'weak_image_raw': image_raw,
                'image': weak_image,
                'density_map': density_map,
                'seg_map': seg_map,
                'all_num': all_num,
                'name': name,
                'is_labeled': is_labeled
            }

        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            return self._get_dummy_sample()

    def _augment(self, image, density_map, seg_map):
        ""
        H, W = image.shape[:2]


        min_hw = min(H, W)
        if min_hw < self.crop_size:
            ratio = float(self.crop_size) / float(min_hw)
            nh, nw = int(H * ratio), int(W * ratio)

            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)


            original_sum = density_map.sum()
            density_map = cv2.resize(density_map, (nw, nh), interpolation=cv2.INTER_CUBIC)
            if original_sum > 0 and density_map.sum() > 0:
                density_map = density_map * original_sum / density_map.sum()

            seg_map = cv2.resize(seg_map, (nw, nh), interpolation=cv2.INTER_NEAREST)

            H, W = nh, nw

        # 2. Random Crop
        if H >= self.crop_size and W >= self.crop_size:
            i = random.randint(0, H - self.crop_size)
            j = random.randint(0, W - self.crop_size)
            image = image[i:i + self.crop_size, j:j + self.crop_size]
            density_map = density_map[i:i + self.crop_size, j:j + self.crop_size]
            seg_map = seg_map[i:i + self.crop_size, j:j + self.crop_size]
        else:

            image = cv2.resize(image, (self.crop_size, self.crop_size),
                               interpolation=cv2.INTER_CUBIC)
            original_sum = density_map.sum()
            density_map = cv2.resize(density_map, (self.crop_size, self.crop_size),
                                     interpolation=cv2.INTER_CUBIC)
            if original_sum > 0 and density_map.sum() > 0:
                density_map = density_map * original_sum / density_map.sum()
            seg_map = cv2.resize(seg_map, (self.crop_size, self.crop_size),
                                 interpolation=cv2.INTER_NEAREST)


        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            density_map = np.fliplr(density_map).copy()
            seg_map = np.fliplr(seg_map).copy()

        return image, density_map, seg_map

    def _strong_augment(self, image):
        ""
        # ColorJitter with p=0.8
        if random.random() < 0.8:

            brightness_factor = 1.0 + (random.random() - 0.5) * 0.8  # ±0.4

            contrast_factor = 1.0 + (random.random() - 0.5) * 0.8  # ±0.4

            image = image * contrast_factor + (brightness_factor - 1.0)
            image = torch.clamp(image, -2.5, 2.5)

        # RandomGrayscale with p=0.2
        if random.random() < 0.2:

            gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            image = gray.unsqueeze(0).repeat(3, 1, 1)

        return image

    def _pad_tensor(self, tensor):
        ""
        if len(tensor.shape) == 3:  # CHW
            _, h, w = tensor.shape
        else:  # HW
            h, w = tensor.shape

        ph = (self.DIV - h % self.DIV) % self.DIV
        pw = (self.DIV - w % self.DIV) % self.DIV

        if ph > 0 or pw > 0:
            pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
            tensor = F.pad(tensor, pad)

        return tensor

    def _get_dummy_sample(self):
        ""
        padded_size = ((self.crop_size + self.DIV - 1) // self.DIV) * self.DIV

        dummy_tensor = torch.zeros(3, padded_size, padded_size)
        dummy_map = torch.zeros(1, padded_size, padded_size)
        dummy_num = torch.zeros(1, 1)

        return {
            'weak_image': dummy_tensor,
            'strong_image': dummy_tensor,
            'weak_image_raw': dummy_tensor,
            'image': dummy_tensor,
            'density_map': dummy_map,
            'seg_map': dummy_map,
            'all_num': dummy_num,
            'name': 'dummy',
            'is_labeled': False
        }


class TwoStreamBatchSampler(Sampler):
    ""

    def __init__(self, labeled_idx, unlabeled_idx, batch_size, num_labeled_per_batch):
        ""
        self.labeled_idx = labeled_idx
        self.unlabeled_idx = unlabeled_idx
        self.batch_size = batch_size
        self.num_labeled = num_labeled_per_batch
        self.num_unlabeled = batch_size - num_labeled_per_batch

        assert len(self.labeled_idx) >= self.num_labeled > 0, \
            f"Not enough labeled samples: {len(self.labeled_idx)} < {self.num_labeled}"

        if len(self.unlabeled_idx) > 0:
            assert len(self.unlabeled_idx) >= self.num_unlabeled >= 0, \
                f"Not enough unlabeled samples: {len(self.unlabeled_idx)} < {self.num_unlabeled}"

    def __iter__(self):
        labeled_iter = self._iterate_once(self.labeled_idx)

        if len(self.unlabeled_idx) > 0 and self.num_unlabeled > 0:
            unlabeled_iter = self._iterate_eternally(self.unlabeled_idx)
            return (
                list(labeled_batch) + list(unlabeled_batch)
                for (labeled_batch, unlabeled_batch) in zip(
                self._grouper(labeled_iter, self.num_labeled),
                self._grouper(unlabeled_iter, self.num_unlabeled)
            )
            )
        else:

            return self._grouper(labeled_iter, self.num_labeled)

    def __len__(self):
        return len(self.labeled_idx) // self.num_labeled

    def _iterate_once(self, indices):
        return iter(np.random.permutation(indices))

    def _iterate_eternally(self, indices):
        ""

        def infinite_shuffles():
            while True:
                yield from np.random.permutation(indices)

        return infinite_shuffles()

    def _grouper(self, iterable, n):
        ""
        args = [iter(iterable)] * n
        return zip(*args)


class ValidationDataset(Dataset):
    ""

    def __init__(self, root, DIV=32):
        self.root = root
        self.im_dir = os.path.join(root, 'images')
        self.gt_dir = os.path.join(root, 'gtdens')
        self.DIV = DIV

        self.rgb_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        self.rgb_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)

        self.filelist = sorted(glob(os.path.join(self.im_dir, '*.jpg')))
        if len(self.filelist) == 0:
            self.filelist = sorted(glob(os.path.join(self.im_dir, '*.png')))

        print(f"ValidationDataset: {len(self.filelist)} images")

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):

        img_path = self.filelist[idx]
        image = io.imread(img_path)

        if len(image.shape) < 3:
            image = np.expand_dims(image, axis=2)
        if image.shape[2] == 1:
            image = np.concatenate((image, image, image), axis=2)


        image = image.astype(np.float32) / 255.0 - self.rgb_mean
        image = image / self.rgb_std


        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        mat_path = os.path.join(self.gt_dir, f'{name}.mat')
        mat = sio.loadmat(mat_path)

        all_num = mat['dot_map'].sum().reshape((1, 1))
        density_map = mat['density_map'].astype(np.float32)
        seg_map = mat['seg_map'].astype(np.float32)


        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        density_map = torch.from_numpy(density_map).unsqueeze(0).float()
        seg_map = torch.from_numpy(seg_map).unsqueeze(0).float()
        all_num = torch.from_numpy(all_num.astype(np.float32))

        # Padding
        image = self._pad_tensor(image)
        density_map = self._pad_tensor(density_map)
        seg_map = self._pad_tensor(seg_map)

        return {
            'image': image,
            'density_map': density_map,
            'seg_map': seg_map,
            'all_num': all_num,
            'name': name
        }

    def _pad_tensor(self, tensor):
        if len(tensor.shape) == 3:
            _, h, w = tensor.shape
        else:
            h, w = tensor.shape

        ph = (self.DIV - h % self.DIV) % self.DIV
        pw = (self.DIV - w % self.DIV) % self.DIV

        if ph > 0 or pw > 0:
            pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
            tensor = F.pad(tensor, pad)

        return tensor


def create_semi_supervised_dataloaders(opt, labeled_ratio=0.1, num_labeled_per_batch=4,
                                       labeled_list_path=None):
    ""
    train_root = os.path.join(opt['sdroot'], 'train')


    train_dataset = SemiSupervisedCountingDataset(
        root=train_root,
        labeled_list_path=labeled_list_path,
        DIV=opt['W2DOpt']['DenPatchSize'],
        crop_size=opt.get('crop_size', 512),
        is_training=True,
        labeled_ratio=labeled_ratio if labeled_list_path is None else None
    )


    if len(train_dataset.unlabeled_idx) == 0:

        print("全监督模式：所有数据都有标签")
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt['bs'],
            shuffle=True,
            num_workers=opt.get('num_workers', 4),
            drop_last=True
        )
    else:

        batch_sampler = TwoStreamBatchSampler(
            labeled_idx=train_dataset.labeled_idx,
            unlabeled_idx=train_dataset.unlabeled_idx,
            batch_size=opt['bs'],
            num_labeled_per_batch=num_labeled_per_batch
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=opt.get('num_workers', 4)
        )


    # val_root = os.path.join(opt['sdroot'], 'test')
    # if not os.path.exists(val_root):
    #     val_root = os.path.join(opt['sdroot'], 'val')
    val_root = os.path.join(opt['sdroot'], 'val')

    val_dataset = ValidationDataset(
        root=val_root,
        DIV=opt['W2DOpt']['DenPatchSize']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=opt.get('num_workers', 4)
    )

    return train_loader, val_loader


if __name__ == '__main__':
    print("Semi-supervised dataset module loaded successfully")
