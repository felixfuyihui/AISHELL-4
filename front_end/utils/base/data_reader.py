#!/usr/bin/env python -u
# -*- coding: utf-8 -*-


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys

import torch
import numpy as np

sys.path.append(os.path.dirname(sys.path[0]) + '/utils')
from base.misc import check, read_path, read_key
from sigproc.sigproc import wavread

class DataReader(object):
    """Data reader for evaluation."""

    def __init__(self, mix_c1_scp):
        """Initialize DataReader. (2 mixtures)

        Args:
            mix_scp: scp file for mixed waves (KALDI format)
        """
        self.key = read_key(mix_c1_scp)
        self.mix_c1_path = read_path(mix_c1_scp)

    def __len__(self):
        return len(self.mix_c1_path)

    def read(self):
        for i in range(len(self.mix_c1_path)):
            key = self.key[i]
            mix_sample = wavread(self.mix_c1_path[i])[0]
            max_norm = np.max(np.abs(mix_sample))
            mix_sample = mix_sample / max_norm
            sample = {
                'key': key,
                'mix': torch.from_numpy(mix_sample.transpose(1,0)).unsqueeze(0),
                'max_norm': max_norm
            }
            yield sample
