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
