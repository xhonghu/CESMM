import os
import cv2
import sys
import pdb
import pickle
import glob
import time
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import torch.utils.data as data
from utils import video_augmentation
from Tokenizer import GlossTokenizer_S2G
from utils import clean_phoenix_2014_trans, clean_phoenix_2014, clean_csl
sys.path.append("..")

def load_dataset_file(filename):
    with open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_tokenizer, dataset='phoenix2014', keypoint='', mode="train"):
        self.mode = mode
        self.clip_len = 300
        if self.mode == 'train':
            self.tmin, self.tmax = 0.5, 1.5
        else:
            self.tmin, self.tmax = 1, 1

        keypoint_path = keypoint + '.' + mode
        self.prefix = prefix
        self.dataset = dataset

        if self.dataset=='CSL-Daily':
            self.w, self.h = 512, 512
        else :
            self.w, self.h = 210, 260

        self.raw_data = load_dataset_file(keypoint_path)

        self.inputs_list = np.load(f"./preprocess/{dataset}/{mode}_info.npy", allow_pickle=True).item()
        self.gloss_tokenizer = GlossTokenizer_S2G(gloss_tokenizer)
        print(mode, len(self))

    def __getitem__(self, idx):
        fi = self.inputs_list[idx]
        directory_path = os.path.dirname(fi['folder'])
        if self.dataset == 'phoenix2014':
            directory_path = directory_path + "/"
        keypoint = self.raw_data[directory_path]['keypoint'].permute(2, 0, 1).to(torch.float32)
        length = keypoint.shape[1]
        if self.dataset=='phoenix2014':
            fi['label'] = fi['label']
        if self.dataset=='phoenix2014-T':
            fi['label'] = clean_phoenix_2014_trans(fi['label'])
        if self.dataset=='CSL-Daily':
            fi['label'] = clean_csl(fi['label']).lower()    # for APP PPT ...,gloss2id is app ppt
        gloss = fi['label']

        return keypoint, gloss, length, fi
    
    def augment_preprocess_inputs(self, keypoints):
        """
        Applies augmentation and preprocessing to keypoint data.
        """
        # 1. 预处理：归一化、Y轴翻转、中心化
        keypoints[:, 0, :, :] /= self.w
        keypoints[:, 1, :, :] = self.h - keypoints[:, 1, :, :]
        keypoints[:, 1, :, :] /= self.h
        keypoints[:, :2, :, :] = (keypoints[:, :2, :, :] - 0.5) / 0.5

        # 2. 仅在训练模式下应用数据增强
        if self.mode == 'train':
            data_numpy = keypoints[:, :2, :, :].permute(0, 2, 3, 1).numpy()
            data_numpy = self.random_translate(data_numpy, p=0.5)
            
            degrees = np.random.uniform(-15, 15)
            theta = np.radians(degrees)
            if np.random.uniform(0, 1) >= 0.5:
                data_numpy = self.rotate_points(data_numpy, theta)

            keypoints[:, :2, :, :] = torch.from_numpy(data_numpy).permute(0, 3, 1, 2)

        return keypoints
    
    def rotate_points(self, points, angle):
        center = [0, 0]
        points_centered = points - center
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        points_rotated = np.dot(points_centered, rotation_matrix.T)
        points_transformed = points_rotated + center
        return points_transformed

    def get_selected_index(self, vlen):
        if self.tmin == 1 and self.tmax == 1:
            if vlen <= self.clip_len:
                frame_index = np.arange(vlen)
            else:
                start = (vlen - self.clip_len) // 2
                frame_index = np.arange(start, start + self.clip_len)
            valid_len = len(frame_index)
        else:
            min_len = min(int(self.tmin * vlen), self.clip_len)
            max_len = min(self.clip_len, int(self.tmax * vlen))
            selected_len = np.random.randint(min_len, max_len + 1)
            if selected_len <= vlen:
                selected_index = sorted(np.random.permutation(np.arange(vlen))[:selected_len])
            else:
                copied_index = np.random.randint(0, vlen, selected_len - vlen)
                selected_index = sorted(np.concatenate([np.arange(vlen), copied_index]))
            frame_index = selected_index
            valid_len = selected_len
        if valid_len % 4 != 0:
            valid_len -= (valid_len % 4)
            frame_index = frame_index[:valid_len]
        assert len(frame_index) == valid_len
        return frame_index, valid_len
    
    def random_translate(self, data_numpy, p=0.5, translate_range=(-0.1, 0.1)):
        """
        随机平移骨架。
        p: 执行平移的概率。
        translate_range: 平移范围（相对于归一化坐标）。
        """
        if np.random.random() < p:
            # data_numpy 的形状是 (B, T, V, C)
            batch_size = data_numpy.shape[0]
            
            # 正确的形状：dx 和 dy 应该是 (B, 1, 1)，这样可以广播到 (B, T, V)
            dx = np.random.uniform(translate_range[0], translate_range[1], size=(batch_size, 1, 1))
            dy = np.random.uniform(translate_range[0], translate_range[1], size=(batch_size, 1, 1))
            
            # data_numpy[..., 0] 的形状是 (B, T, V)
            # dx 的形状 (B, 1, 1) 可以正确广播到 (B, T, V)
            data_numpy[..., 0] += dx
            data_numpy[..., 1] += dy
        return data_numpy
    
    def collate_fn(self, batch):
        keypoint_batch, gloss_batch, src_length_batch, info_batch = [], [], [], []
        for keypoint_sample, gloss_sample, length, info in batch:
            index, valid_len = self.get_selected_index(length)
            keypoint_batch.append(torch.stack([keypoint_sample[:, i, :] for i in index], dim=1))
            src_length_batch.append(valid_len)
            gloss_batch.append(gloss_sample)
            info_batch.append(info)

        # Pad keypoint sequences to the maximum length in the batch
        max_length = max(src_length_batch)
        padded_sgn_keypoints = []
        for keypoints, len_ in zip(keypoint_batch, src_length_batch):
            if len_ < max_length:
                padding = keypoints[:, -1, :].unsqueeze(1)
                padding = torch.tile(padding, [1, max_length - len_, 1])
                padded_keypoint = torch.cat([keypoints, padding], dim=1)
                padded_sgn_keypoints.append(padded_keypoint)
            else:
                padded_sgn_keypoints.append(keypoints)

        keypoints = torch.stack(padded_sgn_keypoints, dim=0)
        keypoints = self.augment_preprocess_inputs(keypoints)

         # Calculate new sequence lengths after potential downsampling by the model
        src_length_batch = torch.LongTensor(src_length_batch)  
        new_src_lengths = (((src_length_batch - 1) / 2) + 1).long()
        new_src_lengths = (((new_src_lengths - 1) / 2) + 1).long()

        # Tokenize glosses
        gloss_input = self.gloss_tokenizer(gloss_batch)   


        label_length = gloss_input['gls_lengths']
        label = gloss_input['gloss_labels']
        return keypoints, new_src_lengths, label, label_length, info_batch

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def __str__(self):
        return f'#total {self.phase} set: {len(self.list)}.'


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()