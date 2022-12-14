# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from ss_baselines.av_exp.models.visual_cnn import conv_output_dim, layer_init

class RIRNet(nn.Module):
    def __init__(self) -> None:
        super(RIRNet, self).__init__()
        self.vnet = nn.Conv1d(2 * 26, 2 * 26, 65)
        self.hnet = nn.Conv1d(2, 64, 26)
        self.flat = nn.Flatten()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(b, c * w, h)
        x = self.vnet(x)
        x = x.reshape(b, c, w)
        x = self.hnet(x)
        x = self.flat(x)
        return x

class AudioEncoder():
    def __init__(self, audiogoal_sensor):
        self._audiogoal_sensor = audiogoal_sensor

    def compute_volume(self, audiogoal):
        num_frame = 150
        nonzero_idx = ((audiogoal > 0.1 * audiogoal.max()).argmax(axis=1))
        rms = np.zeros(nonzero_idx.shape)
        for i in range(len(nonzero_idx)):
            impulse = audiogoal[i, nonzero_idx[i]: nonzero_idx[i] + num_frame]
            rms[i] = np.mean(impulse ** 2)
        return rms

    def __call__(self, observations):
        cnn_input = []

        audio_observations = observations[self._audiogoal_sensor]
        cnn_input.append(audio_observations)

        cnn_input = torch.cat(cnn_input, dim=1).detach().cpu().numpy()
        process = cnn_input.shape[0]
        cnn_input = cnn_input.reshape(-1, cnn_input.shape[-1])
        rms = self.compute_volume(cnn_input)
        rms = rms.reshape(process, -1)

        return torch.from_numpy(rms).cuda().float()


class AudioCNN(nn.Module):
    r"""A Simple 3-Conv CNN for processing audio spectrogram features

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
    """

    def __init__(self, observation_space, output_size, audiogoal_sensor):
        super(AudioCNN, self).__init__()
        audios = observation_space.spaces[audiogoal_sensor]
        self._n_input_audio = audios.shape[2]
        self._audiogoal_sensor = audiogoal_sensor

        cnn_dims = np.array(
            audios.shape[:2], dtype=np.float32
        )

        if cnn_dims[0] < 30 or cnn_dims[1] < 30:
            self._cnn_layers_kernel_size = [(5, 5), (3, 3), (3, 3)]
            self._cnn_layers_stride = [(2, 2), (2, 2), (1, 1)]
        else:
            self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
            self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]

        for kernel_size, stride in zip(
            self._cnn_layers_kernel_size, self._cnn_layers_stride
        ):
            cnn_dims = conv_output_dim(
                dimension=cnn_dims,
                padding=np.array([0, 0], dtype=np.float32),
                dilation=np.array([1, 1], dtype=np.float32),
                kernel_size=np.array(kernel_size, dtype=np.float32),
                stride=np.array(stride, dtype=np.float32),
            )

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self._n_input_audio,
                out_channels=32,
                kernel_size=self._cnn_layers_kernel_size[0],
                stride=self._cnn_layers_stride[0],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[1],
                stride=self._cnn_layers_stride[1],
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=self._cnn_layers_kernel_size[2],
                stride=self._cnn_layers_stride[2],
            ),
            #  nn.ReLU(True),
            Flatten(),
            nn.Linear(64 * cnn_dims[0] * cnn_dims[1], output_size),
            nn.ReLU(True),
        )

        layer_init(self.cnn)

    def forward(self, observations, rir = None, penultimate = False):
        cnn_input = []
        if rir is not None:
            audio_observations = rir
        else:
            audio_observations = observations[self._audiogoal_sensor]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            audio_observations = audio_observations.permute(0, 3, 1, 2)

        cnn_input.append(audio_observations)

        cnn_input = torch.cat(cnn_input, dim=1)
        if penultimate:
            feature = self.cnn[:-2](cnn_input)
            return self.cnn[-2:](feature), feature
        return self.cnn(cnn_input)
