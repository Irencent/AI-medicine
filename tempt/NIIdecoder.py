import io
import os
import os.path as osp
import shutil
import warnings

import mmcv
import numpy as np
import torch
from mmcv.fileio import FileClient
from torch.nn.modules.utils import _pair

from ...utils import get_random_string, get_shm_dir, get_thread_id
from ..builder import PIPELINES
import random
import cv2
import matplotlib.pyplot as plt
import SimpleITK as sitk
import pickle as pkl


@PIPELINES.register_module()
class LoadHVULabel:
    """Convert the HVU label from dictionaries to torch tensors.

    Required keys are "label", "categories", "category_nums", added or modified
    keys are "label", "mask" and "category_mask".
    """

    def __init__(self, **kwargs):
        self.hvu_initialized = False
        self.kwargs = kwargs

    def init_hvu_info(self, categories, category_nums):
        assert len(categories) == len(category_nums)
        self.categories = categories
        self.category_nums = category_nums
        self.num_categories = len(self.categories)
        self.num_tags = sum(self.category_nums)
        self.category2num = dict(zip(categories, category_nums))
        self.start_idx = [0]
        for i in range(self.num_categories - 1):
            self.start_idx.append(self.start_idx[-1] + self.category_nums[i])
        self.category2startidx = dict(zip(categories, self.start_idx))
        self.hvu_initialized = True

    def __call__(self, results):
        """Convert the label dictionary to 3 tensors: "label", "mask" and
        "category_mask".

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        if not self.hvu_initialized:
            self.init_hvu_info(results['categories'], results['category_nums'])

        onehot = torch.zeros(self.num_tags)
        onehot_mask = torch.zeros(self.num_tags)
        category_mask = torch.zeros(self.num_categories)

        for category, tags in results['label'].items():
            category_mask[self.categories.index(category)] = 1.
            start_idx = self.category2startidx[category]
            category_num = self.category2num[category]
            tags = [idx + start_idx for idx in tags]
            onehot[tags] = 1.
            onehot_mask[start_idx:category_num + start_idx] = 1.

        results['label'] = onehot
        results['mask'] = onehot_mask
        results['category_mask'] = category_mask
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'hvu_initialized={self.hvu_initialized})')
        return repr_str


@PIPELINES.register_module()
class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 frame_uniform=False,
                 lge_clip_len=None,
                 lge_frame_interval=None,
                 lge_num_clips=None,
                 frmshift=True):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.lge_clip_len = lge_clip_len
        self.lge_frame_interval = lge_frame_interval
        self.lge_num_clips = lge_num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.frame_uniform = frame_uniform
        self.frmshift = frmshift
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames, lge=False):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        if lge:
            ori_clip_len = self.lge_clip_len * self.lge_frame_interval
            avg_interval = (num_frames - ori_clip_len +
                            1) // self.lge_num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.lge_num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.lge_num_clips)
            elif num_frames > max(self.lge_num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.lge_num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.lge_num_clips
                clip_offsets = np.around(np.arange(self.lge_num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.lge_num_clips, ), dtype=np.int)
        else:
            ori_clip_len = self.clip_len * self.frame_interval
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames, lge=False):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        if lge:
            ori_clip_len = self.lge_clip_len * self.lge_frame_interval
            avg_interval = (num_frames - ori_clip_len + 1) / \
                float(self.lge_num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.lge_num_clips) * avg_interval
                clip_offsets = (base_offsets + avg_interval /
                                2.0).astype(np.int)
                if self.twice_sample:
                    clip_offsets = np.concatenate([clip_offsets, base_offsets])
            else:
                clip_offsets = np.zeros((self.lge_num_clips, ), dtype=np.int)
        else:
            ori_clip_len = self.clip_len * self.frame_interval
            avg_interval = (num_frames - ori_clip_len + 1) / \
                float(self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + avg_interval /
                                2.0).astype(np.int)
                if self.twice_sample:
                    clip_offsets = np.concatenate([clip_offsets, base_offsets])
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames, lge=False):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames, lge)
        else:
            clip_offsets = self._get_train_clips(num_frames, lge)

        return clip_offsets

    def get_seq_frames(self, num_frames, lge=False):
        """
        Modified from https://github.com/facebookresearch/SlowFast/blob/64abcc90ccfdcbb11cf91d6e525bed60e92a8796/slowfast/datasets/ssv2.py#L159
        Given the video index, return the list of sampled frame indexes.
        Args:
            num_frames (int): Total number of frame in the video.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """

        if lge:
            seg_size = float(num_frames - 1) / self.lge_clip_len
            seq = []
            for i in range(self.lge_clip_len):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                if not self.test_mode:
                    seq.append(random.randint(start, end))
                else:
                    seq.append((start + end) // 2)
        else:
            seg_size = float(num_frames - 1) / self.clip_len
            seq = []
            for i in range(self.clip_len):
                start = int(np.round(seg_size * i))
                end = int(np.round(seg_size * (i + 1)))
                if not self.test_mode:
                    seq.append(random.randint(start, end))
                else:
                    seq.append((start + end) // 2)

        return np.array(seq)

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'type' in results:
            type = results['type']
        else:
            type = None
        if results['fusion']:
            if 'sax' in type or '4ch' in type:
                total_frames_cine = results['total_frames_cine']
                if self.frame_uniform:
                    assert results['start_index'] == 0
                    frame_inds_cine = self.get_seq_frames(total_frames_cine)
                else:
                    clip_offsets = self._sample_clips(total_frames_cine)
                    frame_inds = clip_offsets[:, None] + np.arange(
                        self.clip_len)[None, :] * self.frame_interval
                    frame_inds_cine = np.concatenate(frame_inds)

                    if self.temporal_jitter:
                        perframe_offsets = np.random.randint(
                            self.frame_interval, size=len(frame_inds_cine))
                        frame_inds_cine += perframe_offsets

                    frame_inds_cine = frame_inds_cine.reshape(
                        (-1, self.clip_len))
                    if self.out_of_bound_opt == 'loop':
                        frame_inds_cine = np.mod(
                            frame_inds_cine, total_frames_cine)
                    elif self.out_of_bound_opt == 'repeat_last':
                        safe_inds = frame_inds_cine < total_frames_cine
                        unsafe_inds = 1 - safe_inds
                        last_ind = np.max(safe_inds * frame_inds_cine, axis=1)
                        new_inds = (safe_inds * frame_inds_cine +
                                    (unsafe_inds.T * last_ind).T)
                        frame_inds_cine = new_inds
                    else:
                        raise ValueError('Illegal out_of_bound option.')

                    start_index = results['start_index']
                    frame_inds_cine = np.concatenate(
                        frame_inds_cine) + start_index

                    results['frame_inds_cine'] = frame_inds_cine.astype(np.int)
                    results['clip_len_cine'] = self.clip_len
                    results['frame_interval_cine'] = self.frame_interval
                    results['num_clips_cine'] = self.num_clips

            if 'lge' in type:
                total_frames_lge = results['total_frames_lge']
                if self.frame_uniform:
                    assert results['start_index'] == 0
                    frame_inds_lge = self.get_seq_frames(
                        total_frames_lge, True)
                else:
                    clip_offsets = self._sample_clips(total_frames_lge, True)
                    frame_inds = clip_offsets[:, None] + np.arange(
                        self.lge_clip_len)[None, :] * self.lge_frame_interval
                    frame_inds_lge = np.concatenate(frame_inds)

                    if self.temporal_jitter:
                        perframe_offsets = np.random.randint(
                            self.lge_frame_interval, size=len(frame_inds_lge))
                        frame_inds_lge += perframe_offsets

                    frame_inds_lge = frame_inds_lge.reshape(
                        (-1, self.lge_clip_len))
                    if self.out_of_bound_opt == 'loop':
                        frame_inds_lge = np.mod(
                            frame_inds_lge, total_frames_lge)
                    elif self.out_of_bound_opt == 'repeat_last':
                        safe_inds = frame_inds_lge < total_frames_lge
                        unsafe_inds = 1 - safe_inds
                        last_ind = np.max(safe_inds * frame_inds_lge, axis=1)
                        new_inds = (safe_inds * frame_inds_lge +
                                    (unsafe_inds.T * last_ind).T)
                        frame_inds_lge = new_inds
                    else:
                        raise ValueError('Illegal out_of_bound option.')

                    frame_inds_lge = np.concatenate(
                        frame_inds_lge) + start_index

                    results['frame_inds_lge'] = frame_inds_lge.astype(np.int)
                    results['clip_len_lge'] = self.lge_clip_len
                    results['frame_interval_lge'] = self.lge_frame_interval
                    results['num_clips_lge'] = self.lge_num_clips

            return results
        else:
            total_frames = results['total_frames']
            if self.frame_uniform:  # sthv2 sampling strategy
                assert results['start_index'] == 0
                frame_inds = self.get_seq_frames(total_frames)
            else:
                clip_offsets = self._sample_clips(total_frames)
                if not self.frmshift:
                    clip_offsets = np.zeros((1), dtype=np.int)
                frame_inds = clip_offsets[:, None] + np.arange(
                    self.clip_len)[None, :] * self.frame_interval
                frame_inds = np.concatenate(frame_inds)

                if self.temporal_jitter:
                    perframe_offsets = np.random.randint(
                        self.frame_interval, size=len(frame_inds))
                    frame_inds += perframe_offsets

                frame_inds = frame_inds.reshape((-1, self.clip_len))
                if self.out_of_bound_opt == 'loop':
                    frame_inds = np.mod(frame_inds, total_frames)
                elif self.out_of_bound_opt == 'repeat_last':
                    safe_inds = frame_inds < total_frames
                    unsafe_inds = 1 - safe_inds
                    last_ind = np.max(safe_inds * frame_inds, axis=1)
                    new_inds = (safe_inds * frame_inds +
                                (unsafe_inds.T * last_ind).T)
                    frame_inds = new_inds
                else:
                    raise ValueError('Illegal out_of_bound option.')

                start_index = results['start_index']
                frame_inds = np.concatenate(frame_inds) + start_index

            results['frame_inds'] = frame_inds.astype(np.int)
            results['clip_len'] = self.clip_len
            results['frame_interval'] = self.frame_interval
            results['num_clips'] = self.num_clips
            return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class UntrimmedSampleFrames:
    """Sample frames from the untrimmed video.

    Required keys are "filename", "total_frames", added or modified keys are
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): The length of sampled clips. Default: 1.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 16.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
    """

    def __init__(self, clip_len=1, frame_interval=16, start_index=None):

        self.clip_len = clip_len
        self.frame_interval = frame_interval

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']
        start_index = results['start_index']

        clip_centers = np.arange(self.frame_interval // 2, total_frames,
                                 self.frame_interval)
        num_clips = clip_centers.shape[0]
        frame_inds = clip_centers[:, None] + np.arange(
            -(self.clip_len // 2), self.clip_len -
            (self.clip_len // 2))[None, :]
        # clip frame_inds to legal range
        frame_inds = np.clip(frame_inds, 0, total_frames - 1)

        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval})')
        return repr_str


@PIPELINES.register_module()
class DenseSampleFrames(SampleFrames):
    """Select frames from the video by dense sample strategy.

    Required keys are "filename", added or modified keys are "total_frames",
    "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        sample_range (int): Total sample range for dense sample.
            Default: 64.
        num_sample_positions (int): Number of sample start positions, Which is
            only used in test mode. Default: 10. That is to say, by default,
            there are at least 10 clips for one input sample in test mode.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 sample_range=64,
                 num_sample_positions=10,
                 temporal_jitter=False,
                 out_of_bound_opt='loop',
                 test_mode=False):
        super().__init__(
            clip_len,
            frame_interval,
            num_clips,
            temporal_jitter,
            out_of_bound_opt=out_of_bound_opt,
            test_mode=test_mode)
        self.sample_range = sample_range
        self.num_sample_positions = num_sample_positions

    def _get_train_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in train mode.

        It will calculate a sample position and sample interval and set
        start index 0 when sample_pos == 1 or randomly choose from
        [0, sample_pos - 1]. Then it will shift the start index by each
        base offset.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_idx = 0 if sample_position == 1 else np.random.randint(
            0, sample_position - 1)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = (base_offsets + start_idx) % num_frames
        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets by dense sample strategy in test mode.

        It will calculate a sample position and sample interval and evenly
        sample several start indexes as start positions between
        [0, sample_position-1]. Then it will shift each start index by the
        base offsets.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        sample_position = max(1, 1 + num_frames - self.sample_range)
        interval = self.sample_range // self.num_clips
        start_list = np.linspace(
            0, sample_position - 1, num=self.num_sample_positions, dtype=int)
        base_offsets = np.arange(self.num_clips) * interval
        clip_offsets = list()
        for start_idx in start_list:
            clip_offsets.extend((base_offsets + start_idx) % num_frames)
        clip_offsets = np.array(clip_offsets)
        return clip_offsets

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'sample_range={self.sample_range}, '
                    f'num_sample_positions={self.num_sample_positions}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SampleAVAFrames(SampleFrames):

    def __init__(self, clip_len, frame_interval=2, test_mode=False):

        super().__init__(clip_len, frame_interval, test_mode=test_mode)

    def _get_clips(self, center_index, skip_offsets, shot_info):
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        if not self.test_mode:
            frame_inds = frame_inds + skip_offsets
        frame_inds = np.clip(frame_inds, shot_info[0], shot_info[1] - 1)
        return frame_inds

    def __call__(self, results):
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        shot_info = results['shot_info']

        center_index = fps * (timestamp - timestamp_start) + 1

        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)

        results['frame_inds'] = np.array(frame_inds, dtype=np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_mode={self.test_mode})')
        return repr_str


@PIPELINES.register_module()
class SampleProposalFrames(SampleFrames):
    """Sample frames from proposals in the video.

    Required keys are "total_frames" and "out_proposals", added or
    modified keys are "frame_inds", "frame_interval", "num_clips",
    'clip_len' and 'num_proposals'.

    Args:
        clip_len (int): Frames of each sampled output clip.
        body_segments (int): Number of segments in course period.
        aug_segments (list[int]): Number of segments in starting and
            ending period.
        aug_ratio (int | float | tuple[int | float]): The ratio
            of the length of augmentation to that of the proposal.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        test_interval (int): Temporal interval of adjacent sampled frames
            in test mode. Default: 6.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        mode (str): Choose 'train', 'val' or 'test' mode.
            Default: 'train'.
    """

    def __init__(self,
                 clip_len,
                 body_segments,
                 aug_segments,
                 aug_ratio,
                 frame_interval=1,
                 test_interval=6,
                 temporal_jitter=False,
                 mode='train'):
        super().__init__(
            clip_len,
            frame_interval=frame_interval,
            temporal_jitter=temporal_jitter)
        self.body_segments = body_segments
        self.aug_segments = aug_segments
        self.aug_ratio = _pair(aug_ratio)
        if not mmcv.is_tuple_of(self.aug_ratio, (int, float)):
            raise TypeError(f'aug_ratio should be int, float'
                            f'or tuple of int and float, '
                            f'but got {type(aug_ratio)}')
        assert len(self.aug_ratio) == 2
        assert mode in ['train', 'val', 'test']
        self.mode = mode
        self.test_interval = test_interval

    @staticmethod
    def _get_train_indices(valid_length, num_segments):
        """Get indices of different stages of proposals in train mode.

        It will calculate the average interval for each segment,
        and randomly shift them within offsets between [0, average_duration].
        If the total number of frames is smaller than num segments, it will
        return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        avg_interval = (valid_length + 1) // num_segments
        if avg_interval > 0:
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = base_offsets + np.random.randint(
                avg_interval, size=num_segments)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    @staticmethod
    def _get_val_indices(valid_length, num_segments):
        """Get indices of different stages of proposals in validation mode.

        It will calculate the average interval for each segment.
        If the total number of valid length is smaller than num segments,
        it will return all zero indices.

        Args:
            valid_length (int): The length of the starting point's
                valid interval.
            num_segments (int): Total number of segments.

        Returns:
            np.ndarray: Sampled frame indices in validation mode.
        """
        if valid_length >= num_segments:
            avg_interval = valid_length / float(num_segments)
            base_offsets = np.arange(num_segments) * avg_interval
            offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
        else:
            offsets = np.zeros((num_segments, ), dtype=np.int)

        return offsets

    def _get_proposal_clips(self, proposal, num_frames):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices in the proposal's three
        stages: starting, course and ending stage.

        Args:
            proposal (obj): The proposal object.
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        # proposal interval: [start_frame, end_frame)
        start_frame = proposal.start_frame
        end_frame = proposal.end_frame
        ori_clip_len = self.clip_len * self.frame_interval

        duration = end_frame - start_frame
        assert duration != 0
        valid_length = duration - ori_clip_len

        valid_starting = max(0,
                             start_frame - int(duration * self.aug_ratio[0]))
        valid_ending = min(num_frames - ori_clip_len + 1,
                           end_frame - 1 + int(duration * self.aug_ratio[1]))

        valid_starting_length = start_frame - valid_starting - ori_clip_len
        valid_ending_length = (valid_ending - end_frame + 1) - ori_clip_len

        if self.mode == 'train':
            starting_offsets = self._get_train_indices(valid_starting_length,
                                                       self.aug_segments[0])
            course_offsets = self._get_train_indices(valid_length,
                                                     self.body_segments)
            ending_offsets = self._get_train_indices(valid_ending_length,
                                                     self.aug_segments[1])
        elif self.mode == 'val':
            starting_offsets = self._get_val_indices(valid_starting_length,
                                                     self.aug_segments[0])
            course_offsets = self._get_val_indices(valid_length,
                                                   self.body_segments)
            ending_offsets = self._get_val_indices(valid_ending_length,
                                                   self.aug_segments[1])
        starting_offsets += valid_starting
        course_offsets += start_frame
        ending_offsets += end_frame

        offsets = np.concatenate(
            (starting_offsets, course_offsets, ending_offsets))
        return offsets

    def _get_train_clips(self, num_frames, proposals):
        """Get clip offsets in train mode.

        It will calculate sampled frame indices of each proposal, and then
        assemble them.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list): Proposals fetched.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        clip_offsets = []
        for proposal in proposals:
            proposal_clip_offsets = self._get_proposal_clips(
                proposal[0][1], num_frames)
            clip_offsets = np.concatenate(
                [clip_offsets, proposal_clip_offsets])

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        It will calculate sampled frame indices based on test interval.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        return np.arange(
            0, num_frames - ori_clip_len, self.test_interval, dtype=np.int)

    def _sample_clips(self, num_frames, proposals):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.
            proposals (list | None): Proposals fetched.
                It is set to None in test mode.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.mode == 'test':
            clip_offsets = self._get_test_clips(num_frames)
        else:
            assert proposals is not None
            clip_offsets = self._get_train_clips(num_frames, proposals)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        out_proposals = results.get('out_proposals', None)
        clip_offsets = self._sample_clips(total_frames, out_proposals)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        start_index = results['start_index']
        frame_inds = np.mod(frame_inds, total_frames) + start_index

        results['frame_inds'] = np.array(frame_inds).astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = (
            self.body_segments + self.aug_segments[0] + self.aug_segments[1])
        if self.mode in ['train', 'val']:
            results['num_proposals'] = len(results['out_proposals'])

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'body_segments={self.body_segments}, '
                    f'aug_segments={self.aug_segments}, '
                    f'aug_ratio={self.aug_ratio}, '
                    f'frame_interval={self.frame_interval}, '
                    f'test_interval={self.test_interval}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'mode={self.mode})')
        return repr_str


@PIPELINES.register_module()
class PyAVInit:
    """Using pyav to initialize the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "filename",
    added or modified keys are "video_reader", and "total_frames".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        kwargs (dict): Args for file client.
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the PyAV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import av
        except ImportError:
            raise ImportError('Please run "conda install av -c conda-forge" '
                              'or "pip install av" to install PyAV first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = av.open(file_obj)

        results['video_reader'] = container
        results['total_frames'] = container.streams.video[0].frames

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}(io_backend=disk)'
        return repr_str


@PIPELINES.register_module()
class PyAVDecode:
    """Using pyav to decode the video.

    PyAV: https://github.com/mikeboers/PyAV

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
    """

    def __init__(self, multi_thread=False):
        self.multi_thread = multi_thread

    def __call__(self, results):
        """Perform the PyAV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max indice to make early stop
        max_inds = max(results['frame_inds'])
        i = 0
        for frame in container.decode(video=0):
            if i > max_inds + 1:
                break
            imgs.append(frame.to_rgb().to_ndarray())
            i += 1

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['imgs'] = [imgs[i % len(imgs)] for i in results['frame_inds']]

        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(multi_thread={self.multi_thread})'
        return repr_str


@PIPELINES.register_module()
class PyAVDecodeMotionVector(PyAVDecode):
    """Using pyav to decode the motion vectors from video.

    Reference: https://github.com/PyAV-Org/PyAV/
        blob/main/tests/test_decode.py

    Required keys are "video_reader" and "frame_inds",
    added or modified keys are "motion_vectors", "frame_inds".

    Args:
        multi_thread (bool): If set to True, it will apply multi
            thread processing. Default: False.
    """

    @staticmethod
    def _parse_vectors(mv, vectors, height, width):
        """Parse the returned vectors."""
        (w, h, src_x, src_y, dst_x,
         dst_y) = (vectors['w'], vectors['h'], vectors['src_x'],
                   vectors['src_y'], vectors['dst_x'], vectors['dst_y'])
        val_x = dst_x - src_x
        val_y = dst_y - src_y
        start_x = dst_x - w // 2
        start_y = dst_y - h // 2
        end_x = start_x + w
        end_y = start_y + h
        for sx, ex, sy, ey, vx, vy in zip(start_x, end_x, start_y, end_y,
                                          val_x, val_y):
            if (sx >= 0 and ex < width and sy >= 0 and ey < height):
                mv[sy:ey, sx:ex] = (vx, vy)

        return mv

    def __call__(self, results):
        """Perform the PyAV motion vector decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if self.multi_thread:
            container.streams.video[0].thread_type = 'AUTO'
        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        # set max index to make early stop
        max_idx = max(results['frame_inds'])
        i = 0
        stream = container.streams.video[0]
        codec_context = stream.codec_context
        codec_context.options = {'flags2': '+export_mvs'}
        for packet in container.demux(stream):
            for frame in packet.decode():
                if i > max_idx + 1:
                    break
                i += 1
                height = frame.height
                width = frame.width
                mv = np.zeros((height, width, 2), dtype=np.int8)
                vectors = frame.side_data.get('MOTION_VECTORS')
                if frame.key_frame:
                    # Key frame don't have motion vectors
                    assert vectors is None
                if vectors is not None and len(vectors) > 0:
                    mv = self._parse_vectors(mv, vectors.to_ndarray(), height,
                                             width)
                imgs.append(mv)

        results['video_reader'] = None
        del container

        # the available frame in pyav may be less than its length,
        # which may raise error
        results['motion_vectors'] = np.array(
            [imgs[i % len(imgs)] for i in results['frame_inds']])
        return results


@PIPELINES.register_module()
class DecordInit:
    """Using decord to initialize the video_reader.

    Decord: https://github.com/dmlc/decord

    Required keys are "filename",
    added or modified keys are "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', num_threads=1, **kwargs):
        self.io_backend = io_backend
        self.num_threads = num_threads
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the Decord initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import decord
        except ImportError:
            raise ImportError(
                'Please run "pip install decord" to install Decord first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        file_obj = io.BytesIO(self.file_client.get(results['filename']))
        container = decord.VideoReader(file_obj, num_threads=self.num_threads)
        results['video_reader'] = container
        results['total_frames'] = len(container)
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'num_threads={self.num_threads})')
        return repr_str


@PIPELINES.register_module()
class DecordDecode:
    """Using decord to decode the video.

    Decord: https://github.com/dmlc/decord

    Required keys are "video_reader", "filename" and "frame_inds",
    added or modified keys are "imgs" and "original_shape".
    """

    def __call__(self, results):
        """Perform the Decord decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        frame_inds = results['frame_inds']
        # Generate frame index mapping in order
        frame_dict = {
            idx: container[idx].asnumpy()
            for idx in np.unique(frame_inds)
        }

        imgs = [frame_dict[idx] for idx in frame_inds]

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class OpenCVInit:
    """Using OpenCV to initialize the video_reader.

    Required keys are "filename", added or modified keys are "new_path",
    "video_reader" and "total_frames".
    """

    def __init__(self, io_backend='disk', **kwargs):
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None
        self.tmp_folder = None
        if self.io_backend != 'disk':
            random_string = get_random_string()
            thread_id = get_thread_id()
            self.tmp_folder = osp.join(get_shm_dir(),
                                       f'{random_string}_{thread_id}')
            os.mkdir(self.tmp_folder)

    def __call__(self, results):
        """Perform the OpenCV initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if self.io_backend == 'disk':
            new_path = results['filename']
        else:
            if self.file_client is None:
                self.file_client = FileClient(self.io_backend, **self.kwargs)

            thread_id = get_thread_id()
            # save the file of same thread at the same place
            new_path = osp.join(self.tmp_folder, f'tmp_{thread_id}.mp4')
            with open(new_path, 'wb') as f:
                f.write(self.file_client.get(results['filename']))

        container = mmcv.VideoReader(new_path)
        results['new_path'] = new_path
        results['video_reader'] = container
        results['total_frames'] = len(container)

        return results

    def __del__(self):
        if self.tmp_folder and osp.exists(self.tmp_folder):
            shutil.rmtree(self.tmp_folder)

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend})')
        return repr_str


@PIPELINES.register_module()
class OpenCVDecode:
    """Using OpenCV to decode the video.

    Required keys are "video_reader", "filename" and "frame_inds", added or
    modified keys are "imgs", "img_shape" and "original_shape".
    """

    def __call__(self, results):
        """Perform the OpenCV decoding.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        container = results['video_reader']
        imgs = list()

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        for frame_ind in results['frame_inds']:
            cur_frame = container[frame_ind]
            # last frame may be None in OpenCV
            while isinstance(cur_frame, type(None)):
                frame_ind -= 1
                cur_frame = container[frame_ind]
            imgs.append(cur_frame)

        results['video_reader'] = None
        del container

        imgs = np.array(imgs)
        # The default channel order of OpenCV is BGR, thus we change it to RGB
        imgs = imgs[:, :, :, ::-1]
        results['imgs'] = list(imgs)
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]

        return results


@PIPELINES.register_module()
class RawFrameDecode:
    """
    Author: airscker
    Date: 2022-11-18 23:21:34
    LastEditors: airscker
    LastEditTime: 2023-03-24 09:55:52
    Description: Load and decode frames with given indices.

    Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 

    Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
    added or modified keys are "imgs", "img_shape" and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``RawFrameDecode`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)
        fusion = results['fusion']
        directory = results['frame_dir']
        filename_tmpl = results['filename_tmpl']
        modality = results['modality']
        if 'type' in results:
            type = results['type']
        else:
            type = None

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        if fusion:
            idx = 0
            offset = results.get('offset', 0)
            if 'sax' in type:
                sax = list()
                if results['frame_inds_cine'].ndim != 1:
                    results['frame_inds_cine'] = np.squeeze(
                        results['frame_inds_cine'])
                for frame_idx in results['frame_inds_cine']:
                    frame_idx += offset
                    filepath = osp.join(
                        directory[idx], filename_tmpl.format(frame_idx))
                    img_bytes = self.file_client.get(filepath)
                    cur_frame1 = mmcv.imfrombytes(
                        img_bytes, channel_order='rgb')
                    img_bytes = self.file_client.get(
                        filepath.replace('mid', 'up'))
                    cur_frame2 = mmcv.imfrombytes(
                        img_bytes, channel_order='rgb')
                    img_bytes = self.file_client.get(
                        filepath.replace('mid', 'down'))
                    cur_frame3 = mmcv.imfrombytes(
                        img_bytes, channel_order='rgb')
                    cur_frame = np.zeros(
                        (cur_frame1.shape[0], cur_frame1.shape[1], 3))
                    cur_frame[:, :, 0] = cur_frame2[:, :, 0]
                    cur_frame[:, :, 1] = cur_frame1[:, :, 1]
                    cur_frame[:, :, 2] = cur_frame3[:, :, 2]
                    sax.append(cur_frame)
                results['sax'] = sax
                results['original_shape_sax'] = sax[0].shape[:2]
                results['img_shape_sax'] = sax[0].shape[:2]
                idx += 1
            if '4ch' in type:
                ch = list()
                if results['frame_inds_cine'].ndim != 1:
                    results['frame_inds_cine'] = np.squeeze(
                        results['frame_inds_cine'])
                for frame_idx in results['frame_inds_cine']:
                    frame_idx += offset
                    filepath = osp.join(
                        directory[idx], filename_tmpl.format(frame_idx))
                    img_bytes = self.file_client.get(filepath)
                    cur_frame = mmcv.imfrombytes(
                        img_bytes, channel_order='rgb')
                    ch.append(cur_frame)
                results['ch'] = ch
                results['original_shape_4ch'] = ch[0].shape[:2]
                results['img_shape_4ch'] = ch[0].shape[:2]
                idx += 1
            if 'lge' in type:
                lge = list()
                if results['frame_inds_lge'].ndim != 1:
                    results['frame_inds_lge'] = np.squeeze(
                        results['frame_inds_lge'])
                for frame_idx in results['frame_inds_lge']:
                    frame_idx += offset
                    filepath = osp.join(
                        directory[idx], filename_tmpl.format(frame_idx))
                    img_bytes = self.file_client.get(filepath)
                    cur_frame = mmcv.imfrombytes(
                        img_bytes, channel_order='rgb')
                    lge.append(cur_frame)
                results['lge'] = lge
                results['original_shape_lge'] = lge[0].shape[:2]
                results['img_shape_lge'] = lge[0].shape[:2]

            return results

        else:
            imgs = list()

            if results['frame_inds'].ndim != 1:
                results['frame_inds'] = np.squeeze(results['frame_inds'])

            offset = results.get('offset', 0)

            for frame_idx in results['frame_inds']:
                frame_idx += offset
                if modality == 'RGB':
                    filepath = osp.join(
                        directory, filename_tmpl.format(frame_idx))
                    if 'mid' in filepath:
                        img_bytes = self.file_client.get(filepath)
                        cur_frame1 = mmcv.imfrombytes(
                            img_bytes, channel_order='rgb')
                        img_bytes = self.file_client.get(
                            filepath.replace('mid', 'up'))
                        cur_frame2 = mmcv.imfrombytes(
                            img_bytes, channel_order='rgb')
                        img_bytes = self.file_client.get(
                            filepath.replace('mid', 'down'))
                        cur_frame3 = mmcv.imfrombytes(
                            img_bytes, channel_order='rgb')
                        cur_frame = np.zeros(
                            (cur_frame1.shape[0], cur_frame1.shape[1], 3))
                        cur_frame[:, :, 0] = cur_frame2[:, :, 0]
                        cur_frame[:, :, 1] = cur_frame1[:, :, 0]
                        cur_frame[:, :, 2] = cur_frame3[:, :, 0]

                    else:
                        img_bytes = self.file_client.get(filepath)
                        # Get frame with channel order RGB directly.
                        cur_frame = mmcv.imfrombytes(
                            img_bytes, channel_order='rgb')
                    imgs.append(cur_frame)
                    # print(f'shape: {cur_frame.shape}')
                elif modality == 'Flow':
                    x_filepath = osp.join(directory,
                                          filename_tmpl.format('x', frame_idx))
                    y_filepath = osp.join(directory,
                                          filename_tmpl.format('y', frame_idx))
                    x_img_bytes = self.file_client.get(x_filepath)
                    x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
                    y_img_bytes = self.file_client.get(y_filepath)
                    y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
                    imgs.extend([x_frame, y_frame])
                else:
                    raise NotImplementedError

            results['imgs'] = imgs
            # print(f'img length: {len(imgs)}, sample shape: {imgs[0].shape}')
            results['original_shape'] = imgs[0].shape[:2]
            results['img_shape'] = imgs[0].shape[:2]

            # we resize the gt_bboxes and proposals to their real scale
            if 'gt_bboxes' in results:
                h, w = results['img_shape']
                scale_factor = np.array([w, h, w, h])
                gt_bboxes = results['gt_bboxes']
                gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
                results['gt_bboxes'] = gt_bboxes
                if 'proposals' in results and results['proposals'] is not None:
                    proposals = results['proposals']
                    proposals = (proposals * scale_factor).astype(np.float32)
                    results['proposals'] = proposals

            return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'decoding_backend={self.decoding_backend})')
        return repr_str


@PIPELINES.register_module()
class NIIDecodeV2:
    """
    Author: airscker
    Date: 2023-01-22 13:09:54
    LastEditors: airscker
    LastEditTime: 2023-03-31 01:27:30
    Description: Load and decode Nifti dataset

    Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
    
    No necessarity of giving the file paths of masks, only crop position supported, higher processing efficiency.
    Tips:
        SAX cinema data:
            Must have keywords: `mid`, `up`, `down` in every patients' sax cinema filenames, such as `1148528_ZHANG_SAN/slice_up.nii.gz`, `1148528_ZHANG_SAN/slice_mid.nii.gz`, `1148528_ZHANG_SAN/slice_down.nii.gz`.\n
            Slices' keyword should represent their physical position along the `z` axis, we recommand you to get it by `SimpleITK.Image.GetOrigin()[-1]`.
    Args: 
        mask_ann: The path of the `nifti filepath <-> mask crop position(np.array([x_min,x_max,y_min,y_max]))` hash map, data structure: `dict()`, only support `.pkl` file.
    """

    def __init__(self, mask_ann: str = None, sax_concat=False):
        self.mask_ann = mask_ann
        self.sax_concat = sax_concat
        if self.mask_ann is None:
            print('Data will be loaded directly without cropping,\
                if Nifti data need to be cropped according to masks pls specify the annotation of masks')
        else:
            assert self.mask_ann.endswith(
                '.pkl'), f'Mask annotation file must contains the datapath-maskpath dictionary, and .pkl format file expected, but {self.mask_ann} given.'
#             print(
#                 'Friendly reminding: please make sure the file path of modality SAX is the path of middle slice')
            self.__load_mask_ann()

    def __load_mask_ann(self):
        with open(self.mask_ann, 'rb')as f:
            self.data_mask_map = pkl.load(f)
        f.close()
#         files = list(self.data_mask_map.keys())
#         for i in range(len(files)):
#             if not os.path.exists(files[i]):
#                 self.data_mask_map.pop(files[i])
#         print(
#             f'{len(files)} mask_ann hash mapping given, {len(self.data_mask_map)} maps available')

    def __get_crop_pos(self, file_path):
        try:
            mask_path = self.data_mask_map[file_path]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # ----------------------------------------------------#
            mask = np.flip(np.rot90(mask, axes=[0, 1]), axis=0)
            # ----------------------------------------------------#

            # mask=np.uint8(255*(mask-np.min(mask))/(np.max(mask)-np.min(mask)))
            ret, mask = cv2.threshold(mask, 20, 255, 0)
            contours, hierarchy = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            x, y, h, w = cv2.boundingRect(contours[0])
            # img=img[y:y+h,x:x+w]
            return [y, y+h, x, x+w]
        except KeyError:
            return None
    
    def __contour2area(self, mask, mask_path):
#         mask shape (50, 68, 75)
        # 4 nested circles: 0->1->2->3
        type1 = np.array([[-1,-1,1,-1],[-1,-1,2,0],[-1,-1,3,1],[-1,-1,-1,2]])
        
        area_arr = np.zeros(mask.shape)
        try:
            for idx in range(mask.shape[0]):
                frm = mask[idx,:,:]
                ret, thresh = cv2.threshold(frm, 127, 255, 0)
                contours, hierarchy = cv2.findContours(np.uint8(thresh), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                assert len(contours) == 4
                assert np.all(hierarchy[0] == type1)

                area = np.zeros(thresh.shape,np.uint8)
                area = cv2.drawContours(area,contours,0,(120,120,120),-1)
                area = cv2.drawContours(area,contours,3,(255,255,255),-1)
                area_arr[idx,:,:] = area
        except:
            print('__contour2area error:', mask_path)
            print('Contour num: %s' % len(contours))
            print('Hierarchy: ' + hierarchy)
            area_arr = mask
        return area_arr
            
    
    def __crop(self, file_path: str, mod: str):
        if not file_path.endswith('.nii.gz'):
            file_path += '.nii.gz'
        if mod == 'sax':
            # mid_slice_num = int(file_path.split(
            #     '/')[-1].split('.nii.gz')[0].split('_')[-1])
            sax_mid = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
            ###
            sax_up = sitk.GetArrayFromImage(sitk.ReadImage(file_path.replace('mid', 'up')))
            sax_down = sitk.GetArrayFromImage(sitk.ReadImage(file_path.replace('mid', 'down')))

            if self.mask_ann is not None:
                # crop_pos = self.__get_crop_pos(file_path=file_path)
                crop_pos = self.data_mask_map[file_path]
                sax_mid = self.clip_top_bottom(
                    sax_mid[:, crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]])
                if np.max(sax_mid) == np.min(sax_mid):
                    print('sax norm_range eror: ', file_path)
                if self.sax_concat:
                    sax_down = self.clip_top_bottom(
                       sax_down[:, crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]])
                    sax_up = self.clip_top_bottom(
                       sax_up[:, crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]])
            try:
                if self.sax_concat:
                    sax_fusion = np.array([sax_up, sax_mid, sax_down])          
                else:
                    sax_fusion = np.array([sax_mid, sax_mid, sax_mid])

            except:
                print(sax_up.shape, sax_down.shape, sax_mid.shape)
                print(file_path, '\n', file_path.replace('mid', 'up'),
                      '\n', file_path.replace('mid', 'down'))
                return 0
            sax_fusion = np.moveaxis(sax_fusion, 0, -1)
            return sax_fusion
        elif mod == '4ch' or mod == 'lge':
            data = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
#            with open('/output/example.txt', 'a') as file:
#                line = f'{file_path}\t'
#                file.write(line)
            if self.mask_ann is not None:
                # crop_pos = self.__get_crop_pos(file_path=file_path)
                crop_pos = self.data_mask_map[file_path]
                data = self.clip_top_bottom(data[:, crop_pos[2]:crop_pos[3],
                                                 crop_pos[0]:crop_pos[1]])
                if np.max(data) == np.min(data):
                    print('4ch norm_range eror: ', file_path)
            rgbdata = np.array([data]*3)
            rgbdata = np.moveaxis(rgbdata, 0, -1)
            # rgbdata = np.zeros((len(data), data.shape[1], data.shape[2], 3))
            # for i in range(len(data)):
            #     rgbdata[i] = self.Gray2RGB(self.clip_top_bottom(data=data[i]))
            return rgbdata
        elif mod == 'seg':
            data = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
#             for area seg
#             data =self.__contour2area(data, file_path)

            if self.mask_ann is not None:
                # crop_pos = self.__get_crop_pos(file_path=file_path)
                mask_path = file_path.replace('_CINE_label', '_CINE')
                crop_pos = self.data_mask_map[mask_path]
                data = self.clip_top_bottom(data[:, crop_pos[2]:crop_pos[3],
                                                 crop_pos[0]:crop_pos[1]])
                if np.max(data) == np.min(data):
                    print('seg norm_range eror: ', file_path)
            rgbdata = np.array([data]*3)
            rgbdata = np.moveaxis(rgbdata, 0, -1)
#             print('seg out:', rgbdata.shape, np.min(rgbdata), np.max(rgbdata))
            # rgbdata = np.zeros((len(data), data.shape[1], data.shape[2], 3))
            # for i in range(len(data)):
            #     rgbdata[i] = self.Gray2RGB(self.clip_top_bottom(data=data[i]))
            return rgbdata


    def norm_range(self, data):
        return np.uint8(255.0*(data-np.min(data))/(np.max(data)-np.min(data)))

    def clip_top_bottom(self, data: np.ndarray, scale=0.001):
        arr = np.sort(data.flatten())
        size = len(arr)

        min_value = arr[int(scale * size)]
        max_value = arr[int((1 - scale) * size)]

        data[np.where(data < min_value)] = min_value
        data[np.where(data > max_value)] = max_value

        return self.norm_range(data=data)

    def __call__(self, results):
        """Perform the ``NIIDecodeV2`` to pick frames given indices.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        fusion = results['fusion']
        file_path = results['frame_dir']

        if 'type' in results:
            type = results['type']
        else:
            type = None

        # if self.file_client is None:
        #     self.file_client = FileClient(self.io_backend, **self.kwargs)

        if fusion:
            idx = 0
            # offset = results.get('offset', 0)
            offset = -1
            if 'sax' in type:
                sax = list()
                if results['frame_inds_cine'].ndim != 1:
                    results['frame_inds_cine'] = np.squeeze(
                        results['frame_inds_cine'])
                # print(file_path[idx])
                sax_fusion = self.__crop(
                    file_path=file_path[idx], mod='sax')
                # print('sax', results['frame_inds_cine'], sax_up.shape)
                for frame_idx in results['frame_inds_cine']:
                    frame_idx += offset
                    sax.append(sax_fusion[frame_idx])
                results['sax'] = np.float32(sax)
                results['original_shape_sax'] = sax[0].shape[:2]
                results['img_shape_sax'] = sax[0].shape[:2]
                idx += 1
            if '4ch' in type:
                ch = list()
                if results['frame_inds_cine'].ndim != 1:
                    results['frame_inds_cine'] = np.squeeze(
                        results['frame_inds_cine'])
                lax4ch_data = self.__crop(file_path=file_path[idx], mod='4ch')
                # print('4ch', results['frame_inds_cine'], lax4ch_data.shape)
                for frame_idx in results['frame_inds_cine']:
                    frame_idx += offset
                    ch.append(lax4ch_data[frame_idx])
                    # filepath = osp.join(directory[idx], filename_tmpl.format(frame_idx))
                    # img_bytes = self.file_client.get(filepath)
                    # cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                    # ch.append(cur_frame)
                results['ch'] = np.float32(ch)
                results['original_shape_4ch'] = ch[0].shape[:2]
                results['img_shape_4ch'] = ch[0].shape[:2]
                idx += 1
            if 'lge' in type:
                lge = list()
                if results['frame_inds_lge'].ndim != 1:
                    results['frame_inds_lge'] = np.squeeze(
                        results['frame_inds_lge'])
                lge_data = self.__crop(file_path=file_path[idx], mod='lge')
                for frame_idx in results['frame_inds_lge']:
                    frame_idx += offset
                    lge.append(lge_data[frame_idx])
                results['lge'] = np.float32(lge)
                results['original_shape_lge'] = lge[0].shape[:2]
                results['img_shape_lge'] = lge[0].shape[:2]

            return results

        else:
            imgs = list()
            # offset = results.get('offset', 0)
            offset = -1
#             print(results['frame_inds'])
            if results['frame_inds'].ndim != 1:
                results['frame_inds'] = np.squeeze(results['frame_inds'])
            if 'sax' in file_path or 'SAX' in file_path:
                sax_fusion = self.__crop(
                    file_path=file_path, mod='sax')
                for frame_idx in results['frame_inds']:
                    frame_idx += offset
                    imgs.append(sax_fusion[frame_idx])
            else:
                data = self.__crop(file_path=file_path, mod='4ch')
#                 print(results['frame_inds'])
#                 print(f'file: {file_path}, type: {type}, shape: {data.shape}, shape2: {data[:,:,0].shape}')
                for frame_idx in results['frame_inds']:
#                     data1 = data[frame_idx]
                    try:
                        data1 = data[frame_idx-1]
                    except:
                        print(f'file: {file_path}, type: {type}, shape: {data.shape}, shape2: {data[:,:,0].shape}')
                        print(frame_idx)
                        print(results['frame_inds'])
                    frame_idx += offset
                    imgs.append(data[frame_idx])

            results['imgs'] = np.float32(imgs)
            # print(f'img length: {len(imgs)}, sample shape: {imgs[0].shape}')
            results['original_shape'] = imgs[0].shape[:2]
            results['img_shape'] = imgs[0].shape[:2]
            
            if 'gt_semantic_seg' in results:
                seg_path = results['label']
                segs = list()
                seg_data = self.__crop(file_path=seg_path, mod='seg')
                
                try:
                    for frame_idx in results['frame_inds']:
                        frame_idx += offset
                        segs.append(seg_data[frame_idx])
                except:
                    print('mask path: ',seg_path)
                    print('label shape:', seg_data.shape)
                    
#                 for frame_idx in results['frame_inds']:
#                     frame_idx += offset
#                     segs.append(seg_data[frame_idx])

                results['gt_semantic_seg'] = np.float32(segs)

            # we resize the gt_bboxes and proposals to their real scale
            if 'gt_bboxes' in results:
                h, w = results['img_shape']
                scale_factor = np.array([w, h, w, h])
                gt_bboxes = results['gt_bboxes']
                gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
                results['gt_bboxes'] = gt_bboxes
                if 'proposals' in results and results['proposals'] is not None:
                    proposals = results['proposals']
                    proposals = (proposals * scale_factor).astype(np.float32)
                    results['proposals'] = proposals

            return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}')
        return repr_str


@PIPELINES.register_module()
class ImageDecode:
    """Load and decode images.

    Required key is "filename", added or modified keys are "imgs", "img_shape"
    and "original_shape".

    Args:
        io_backend (str): IO backend where frames are stored. Default: 'disk'.
        decoding_backend (str): Backend used for image decoding.
            Default: 'cv2'.
        kwargs (dict, optional): Arguments for FileClient.
    """

    def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
        self.io_backend = io_backend
        self.decoding_backend = decoding_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        """Perform the ``ImageDecode`` to load image given the file path.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        mmcv.use_backend(self.decoding_backend)

        filename = results['filename']

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)

        imgs = list()
        img_bytes = self.file_client.get(filename)

        img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
        imgs.append(img)

        results['imgs'] = imgs
        results['original_shape'] = imgs[0].shape[:2]
        results['img_shape'] = imgs[0].shape[:2]
        return results


@PIPELINES.register_module()
class AudioDecodeInit:
    """Using librosa to initialize the audio reader.

    Required keys are "audio_path", added or modified keys are "length",
    "sample_rate", "audios".

    Args:
        io_backend (str): io backend where frames are store.
            Default: 'disk'.
        sample_rate (int): Audio sampling times per second. Default: 16000.
    """

    def __init__(self,
                 io_backend='disk',
                 sample_rate=16000,
                 pad_method='zero',
                 **kwargs):
        self.io_backend = io_backend
        self.sample_rate = sample_rate
        if pad_method in ['random', 'zero']:
            self.pad_method = pad_method
        else:
            raise NotImplementedError
        self.kwargs = kwargs
        self.file_client = None

    @staticmethod
    def _zero_pad(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _random_pad(shape):
        # librosa load raw audio file into a distribution of -1~+1
        return np.random.rand(shape).astype(np.float32) * 2 - 1

    def __call__(self, results):
        """Perform the librosa initialization.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        try:
            import librosa
        except ImportError:
            raise ImportError('Please install librosa first.')

        if self.file_client is None:
            self.file_client = FileClient(self.io_backend, **self.kwargs)
        if osp.exists(results['audio_path']):
            file_obj = io.BytesIO(self.file_client.get(results['audio_path']))
            y, sr = librosa.load(file_obj, sr=self.sample_rate)
        else:
            # Generate a random dummy 10s input
            pad_func = getattr(self, f'_{self.pad_method}_pad')
            y = pad_func(int(round(10.0 * self.sample_rate)))
            sr = self.sample_rate

        results['length'] = y.shape[0]
        results['sample_rate'] = sr
        results['audios'] = y
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'io_backend={self.io_backend}, '
                    f'sample_rate={self.sample_rate}, '
                    f'pad_method={self.pad_method})')
        return repr_str


@PIPELINES.register_module()
class LoadAudioFeature:
    """Load offline extracted audio features.

    Required keys are "audio_path", added or modified keys are "length",
    audios".
    """

    def __init__(self, pad_method='zero'):
        if pad_method not in ['zero', 'random']:
            raise NotImplementedError
        self.pad_method = pad_method

    @staticmethod
    def _zero_pad(shape):
        return np.zeros(shape, dtype=np.float32)

    @staticmethod
    def _random_pad(shape):
        # spectrogram is normalized into a distribution of 0~1
        return np.random.rand(shape).astype(np.float32)

    def __call__(self, results):
        """Perform the numpy loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if osp.exists(results['audio_path']):
            feature_map = np.load(results['audio_path'])
        else:
            # Generate a random dummy 10s input
            # Some videos do not have audio stream
            pad_func = getattr(self, f'_{self.pad_method}_pad')
            feature_map = pad_func((640, 80))

        results['length'] = feature_map.shape[0]
        results['audios'] = feature_map
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'pad_method={self.pad_method})')
        return repr_str


@PIPELINES.register_module()
class AudioDecode:
    """Sample the audio w.r.t. the frames selected.

    Args:
        fixed_length (int): As the audio clip selected by frames sampled may
            not be exactly the same, `fixed_length` will truncate or pad them
            into the same size. Default: 32000.

    Required keys are "frame_inds", "num_clips", "total_frames", "length",
    added or modified keys are "audios", "audios_shape".
    """

    def __init__(self, fixed_length=32000):
        self.fixed_length = fixed_length

    def __call__(self, results):
        """Perform the ``AudioDecode`` to pick audio clips."""
        audio = results['audios']
        frame_inds = results['frame_inds']
        num_clips = results['num_clips']
        resampled_clips = list()
        frame_inds = frame_inds.reshape(num_clips, -1)
        for clip_idx in range(num_clips):
            clip_frame_inds = frame_inds[clip_idx]
            start_idx = max(
                0,
                int(
                    round((clip_frame_inds[0] + 1) / results['total_frames'] *
                          results['length'])))
            end_idx = min(
                results['length'],
                int(
                    round((clip_frame_inds[-1] + 1) / results['total_frames'] *
                          results['length'])))
            cropped_audio = audio[start_idx:end_idx]
            if cropped_audio.shape[0] >= self.fixed_length:
                truncated_audio = cropped_audio[:self.fixed_length]
            else:
                truncated_audio = np.pad(
                    cropped_audio,
                    ((0, self.fixed_length - cropped_audio.shape[0])),
                    mode='constant')

            resampled_clips.append(truncated_audio)

        results['audios'] = np.array(resampled_clips)
        results['audios_shape'] = results['audios'].shape
        return results


@PIPELINES.register_module()
class BuildPseudoClip:
    """Build pseudo clips with one single image by repeating it n times.

    Required key is "imgs", added or modified key is "imgs", "num_clips",
        "clip_len".

    Args:
        clip_len (int): Frames of the generated pseudo clips.
    """

    def __init__(self, clip_len):
        self.clip_len = clip_len

    def __call__(self, results):
        # the input should be one single image
        assert len(results['imgs']) == 1
        im = results['imgs'][0]
        for _ in range(1, self.clip_len):
            results['imgs'].append(np.copy(im))
        results['clip_len'] = self.clip_len
        results['num_clips'] = 1
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@PIPELINES.register_module()
class FrameSelector(RawFrameDecode):
    """Deprecated class for ``RawFrameDecode``."""

    def __init__(self, *args, **kwargs):
        warnings.warn('"FrameSelector" is deprecated, please switch to'
                      '"RawFrameDecode"')
        super().__init__(*args, **kwargs)


@PIPELINES.register_module()
class AudioFeatureSelector:
    """Sample the audio feature w.r.t. the frames selected.

    Required keys are "audios", "frame_inds", "num_clips", "length",
    "total_frames", added or modified keys are "audios", "audios_shape".

    Args:
        fixed_length (int): As the features selected by frames sampled may
            not be extactly the same, `fixed_length` will truncate or pad them
            into the same size. Default: 128.
    """

    def __init__(self, fixed_length=128):
        self.fixed_length = fixed_length

    def __call__(self, results):
        """Perform the ``AudioFeatureSelector`` to pick audio feature clips.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        audio = results['audios']
        frame_inds = results['frame_inds']
        num_clips = results['num_clips']
        resampled_clips = list()

        frame_inds = frame_inds.reshape(num_clips, -1)
        for clip_idx in range(num_clips):
            clip_frame_inds = frame_inds[clip_idx]
            start_idx = max(
                0,
                int(
                    round((clip_frame_inds[0] + 1) / results['total_frames'] *
                          results['length'])))
            end_idx = min(
                results['length'],
                int(
                    round((clip_frame_inds[-1] + 1) / results['total_frames'] *
                          results['length'])))
            cropped_audio = audio[start_idx:end_idx, :]
            if cropped_audio.shape[0] >= self.fixed_length:
                truncated_audio = cropped_audio[:self.fixed_length, :]
            else:
                truncated_audio = np.pad(
                    cropped_audio,
                    ((0, self.fixed_length - cropped_audio.shape[0]), (0, 0)),
                    mode='constant')

            resampled_clips.append(truncated_audio)
        results['audios'] = np.array(resampled_clips)
        results['audios_shape'] = results['audios'].shape
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'fix_length={self.fixed_length})')
        return repr_str


@PIPELINES.register_module()
class LoadLocalizationFeature:
    """Load Video features for localizer with given video_name list.

    Required keys are "video_name" and "data_prefix", added or modified keys
    are "raw_feature".

    Args:
        raw_feature_ext (str): Raw feature file extension.  Default: '.csv'.
    """

    def __init__(self, raw_feature_ext='.csv'):
        valid_raw_feature_ext = ('.csv', )
        if raw_feature_ext not in valid_raw_feature_ext:
            raise NotImplementedError
        self.raw_feature_ext = raw_feature_ext

    def __call__(self, results):
        """Perform the LoadLocalizationFeature loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        data_prefix = results['data_prefix']

        data_path = osp.join(data_prefix, video_name + self.raw_feature_ext)
        raw_feature = np.loadtxt(
            data_path, dtype=np.float32, delimiter=',', skiprows=1)

        results['raw_feature'] = np.transpose(raw_feature, (1, 0))

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'raw_feature_ext={self.raw_feature_ext})')
        return repr_str


@PIPELINES.register_module()
class GenerateLocalizationLabels:
    """Load video label for localizer with given video_name list.

    Required keys are "duration_frame", "duration_second", "feature_frame",
    "annotations", added or modified keys are "gt_bbox".
    """

    def __call__(self, results):
        """Perform the GenerateLocalizationLabels loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_frame = results['duration_frame']
        video_second = results['duration_second']
        feature_frame = results['feature_frame']
        corrected_second = float(feature_frame) / video_frame * video_second
        annotations = results['annotations']

        gt_bbox = []

        for annotation in annotations:
            current_start = max(
                min(1, annotation['segment'][0] / corrected_second), 0)
            current_end = max(
                min(1, annotation['segment'][1] / corrected_second), 0)
            gt_bbox.append([current_start, current_end])

        gt_bbox = np.array(gt_bbox)
        results['gt_bbox'] = gt_bbox
        return results


@PIPELINES.register_module()
class LoadProposals:
    """Loading proposals with given proposal results.

    Required keys are "video_name", added or modified keys are 'bsp_feature',
    'tmin', 'tmax', 'tmin_score', 'tmax_score' and 'reference_temporal_iou'.

    Args:
        top_k (int): The top k proposals to be loaded.
        pgm_proposals_dir (str): Directory to load proposals.
        pgm_features_dir (str): Directory to load proposal features.
        proposal_ext (str): Proposal file extension. Default: '.csv'.
        feature_ext (str): Feature file extension. Default: '.npy'.
    """

    def __init__(self,
                 top_k,
                 pgm_proposals_dir,
                 pgm_features_dir,
                 proposal_ext='.csv',
                 feature_ext='.npy'):
        self.top_k = top_k
        self.pgm_proposals_dir = pgm_proposals_dir
        self.pgm_features_dir = pgm_features_dir
        valid_proposal_ext = ('.csv', )
        if proposal_ext not in valid_proposal_ext:
            raise NotImplementedError
        self.proposal_ext = proposal_ext
        valid_feature_ext = ('.npy', )
        if feature_ext not in valid_feature_ext:
            raise NotImplementedError
        self.feature_ext = feature_ext

    def __call__(self, results):
        """Perform the LoadProposals loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        video_name = results['video_name']
        proposal_path = osp.join(self.pgm_proposals_dir,
                                 video_name + self.proposal_ext)
        if self.proposal_ext == '.csv':
            pgm_proposals = np.loadtxt(
                proposal_path, dtype=np.float32, delimiter=',', skiprows=1)

        pgm_proposals = np.array(pgm_proposals[:self.top_k])
        tmin = pgm_proposals[:, 0]
        tmax = pgm_proposals[:, 1]
        tmin_score = pgm_proposals[:, 2]
        tmax_score = pgm_proposals[:, 3]
        reference_temporal_iou = pgm_proposals[:, 5]

        feature_path = osp.join(self.pgm_features_dir,
                                video_name + self.feature_ext)
        if self.feature_ext == '.npy':
            bsp_feature = np.load(feature_path).astype(np.float32)

        bsp_feature = bsp_feature[:self.top_k, :]

        results['bsp_feature'] = bsp_feature
        results['tmin'] = tmin
        results['tmax'] = tmax
        results['tmin_score'] = tmin_score
        results['tmax_score'] = tmax_score
        results['reference_temporal_iou'] = reference_temporal_iou

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'top_k={self.top_k}, '
                    f'pgm_proposals_dir={self.pgm_proposals_dir}, '
                    f'pgm_features_dir={self.pgm_features_dir}, '
                    f'proposal_ext={self.proposal_ext}, '
                    f'feature_ext={self.feature_ext})')
        return repr_str
