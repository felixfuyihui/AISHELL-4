#!/usr/bin/env python -u
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import numpy as np

from torch.utils.data import Dataset

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from base.misc import check, read_path
from sigproc.sigproc import wavread

class TimeDomainDateset(Dataset):
    """Dataset class for time-domian speech separation."""

    def __init__(self,
                 mix_c1_scp,
                 s1_c1_scp,
                 s2_c1_scp,
                 utt2dur,
                 sample_rate=16000,
                 sample_clip_size=4):
        """Initialize the TimeDomainDateset. (2 mixtures)

        Args:
            mix_scp: scp file for mixed waves (KALDI format)
            s1_scp: scp file for speaker 1
            s2_scp: scp file for speaker 2
            sample_clip_size: segmental length (default: 4s)
        """
        check(mix_c1_scp, s1_c1_scp, s2_c1_scp)
        self.sample_rate = sample_rate
        self.sample_clip_size = sample_clip_size
        self.segment_length = self.sample_rate * self.sample_clip_size

        self.mix_c1_path = read_path(mix_c1_scp)
        self.s1_c1_path = read_path(s1_c1_scp)
        self.s2_c1_path = read_path(s2_c1_scp)
        self.utt2dur = read_path(utt2dur)
        


        self.retrieve_index = []
        for i in range(len(self.mix_c1_path)):
            sample_size = int(self.utt2dur[i])
            if sample_size < self.segment_length:
                # wave length is smaller than segmental length
                if sample_size * 2 < self.segment_length:
                    continue
                self.retrieve_index.append((i, -1))
            else:
                # Cut wave into clips and restore the retrieve index
                sample_index = 0
                while sample_index + self.segment_length < sample_size:
                    self.retrieve_index.append((i, sample_index))
                    sample_index += self.segment_length
                if sample_index != sample_size - 1:
                    self.retrieve_index.append(
                            (i, sample_size - self.segment_length))

    def __len__(self):
        return len(self.retrieve_index)

    def __getitem__(self, index):
        utt_id, sample_index = self.retrieve_index[index]
        mix_c1_sample = wavread(self.mix_c1_path[utt_id])[0]
        s1_c1_sample = wavread(self.s1_c1_path[utt_id])[0]
        s2_c1_sample = wavread(self.s2_c1_path[utt_id])[0]
        if len(s1_c1_sample.shape) > 1:
            s1_c1_sample = s1_c1_sample[:,0]
        if len(s2_c1_sample.shape) > 1:
            s2_c1_sample = s2_c1_sample[:,0]
        try:
            whitenoise = np.random.normal(0,1,mix_c1_sample.shape).astype(np.float32) * 5e-4
            mix_c1_sample = mix_c1_sample + whitenoise
            s1_c1_sample = s1_c1_sample + whitenoise[:,0]
            s2_c1_sample = s2_c1_sample + whitenoise[:,0]
        except:
            whitenoise = np.random.normal(0,1,mix_c1_sample.shape).astype(np.float32) * 5e-4
            mix_c1_sample = whitenoise
            s1_c1_sample = whitenoise[:,0]
            s2_c1_sample = whitenoise[:,0]
            print(self.mix_c1_path[utt_id])
            print(self.s1_c1_path[utt_id])
            print(self.s2_c1_path[utt_id])
        if sample_index == -1:
            length = len(mix_c1_sample)
            stack_length = self.segment_length - length
            mix_c1_stack_sample = mix_c1_sample[: stack_length]
            s1_c1_stack_sample = s1_c1_sample[: stack_length]
            s2_c1_stack_sample = s2_c1_sample[: stack_length]

            mix_c1_clipped_sample = np.concatenate(
                    (mix_c1_sample, mix_c1_stack_sample), axis=0)
            s1_c1_clipped_sample = np.concatenate(
                    (s1_c1_sample, s1_c1_stack_sample), axis=0)
            s2_c1_clipped_sample = np.concatenate(
                    (s2_c1_sample, s2_c1_stack_sample), axis=0)
        else:
            end_index = sample_index + self.segment_length
            mix_c1_clipped_sample = mix_c1_sample[sample_index : end_index]
            s1_c1_clipped_sample = s1_c1_sample[sample_index : end_index]
            s2_c1_clipped_sample = s2_c1_sample[sample_index : end_index]

        mix_clipped_sample = mix_c1_clipped_sample
        src_clipped_sample = np.stack(
            (s1_c1_clipped_sample,
             s2_c1_clipped_sample
            ), axis=0)
        sample = {
            'mix': mix_clipped_sample.transpose(1,0),
            'src': src_clipped_sample,
        }
        return sample
