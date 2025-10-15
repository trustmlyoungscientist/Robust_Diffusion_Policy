from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class PushTLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', obs_key, state_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        # start_id = max(idx-128,0)
        # import cv2
        # for i in range(128): # 创建一个 4x4 的子图，共16个子图
        #     # fig.suptitle(f'Sample {i+1}', fontsize=16)

        #     # for j in range(9):
        #     # ax = axes[j // 4, j % 4]
        #     coords = self.replay_buffer.data['keypoint'][start_id+i, :].reshape(9, 2)
        #     img = self.replay_buffer.data['img'][start_id+i, :]
        #     coords_action = self.replay_buffer.data['action'][start_id+i, :].reshape(1, 2)

        #     scale_x = 96 / 500
        #     scale_y = 96 / 500

        #     # 将 coords 从原始坐标系缩放到新的坐标系
        #     scaled_coords = coords * [scale_x, scale_y]
        #     scaled_coords_action = coords_action * [scale_x, scale_y]

        #     for coord in scaled_coords:
        #         cv2.circle(img, tuple(coord.astype(int)), 2, (0, 0, 255), -1)  # 绘制红色圆点表示关键点
            
        #     for coord in scaled_coords_action:
        #         cv2.circle(img, tuple(coord.astype(int)), 2, (255, 0, 0), -1)
        #     # 绘制动作点为星形
        #     # cv2.drawMarker(img, tuple(coords_action.astype(int)), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=10, thickness=2)  # 绘制绿色星形标记
        #     cv2.imwrite(f'{i}.jpg', img)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
