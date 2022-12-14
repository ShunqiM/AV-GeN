#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import abc
from random import vonmisesvariate
from librosa.core import audio
from numpy.testing._private.utils import assert_

import torch
import torch.nn as nn
from torchaudio import kaldi_io
from torchsummary import summary

from ss_baselines.common.utils import CategoricalNet
from ss_baselines.av_exp.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.av_exp.models.visual_cnn import VisualCNN
from ss_baselines.av_exp.models.audio_cnn import AudioCNN

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        features, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        return self.critic(features)

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action, ssound = False
    ):
        if ssound:
            features, rnn_hidden_states, predicted, target = self.net(
                observations, rnn_hidden_states, prev_actions, masks, synthesis=True
            )
        else:
            features, rnn_hidden_states= self.net(
                observations, rnn_hidden_states, prev_actions, masks
            )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()
        if ssound:
            return value, action_log_probs, distribution_entropy, rnn_hidden_states, predicted, target
        return value, action_log_probs, distribution_entropy, rnn_hidden_states


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False,
        pretrained=False,
        ssound = False, 
        au_width = 10,
        batch = 32
    ):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb,
                pretrained=pretrained,
                ssound = ssound, 
                au_width = au_width,
                batch=batch
            ),
            action_space.n,
        )


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    For feature contrastive learning, it also computes the forward pass results for the contrastive batches 
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False, pretrained=False, ssound = False, au_width = 10, batch=32):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0
        self._ssound = ssound
        self.batch_size = batch

        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True


        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        if self._audiogoal:
            if 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'
            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)


        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + (self._hidden_size if self._audiogoal else 0)
        self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)

        if 'rgb' in observation_space.spaces and not extra_rgb:
            rgb_shape = observation_space.spaces['rgb'].shape
            summary(self.visual_encoder.cnn, (rgb_shape[2], rgb_shape[0], rgb_shape[1]), device='cpu')
        if 'depth' in observation_space.spaces:
            depth_shape = observation_space.spaces['depth'].shape
            summary(self.visual_encoder.cnn, (depth_shape[2], depth_shape[0], depth_shape[1]), device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')
        
        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, synthesis=False):
        x = []
        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            audio_feature = self.audio_encoder(observations)
            x.append(audio_feature)
        if not self.is_blind:
            x.append(self.visual_encoder(observations))

        x1 = torch.cat(x, dim=1)

        x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        if self._ssound and synthesis:
            audio_batch, target_batch, audio_feature = sample_batch(observations['spectrogram'], observations['switch_spectrogram'], audio_feature, self.batch_size)
            observations['spectrogram'] = target_batch.permute(0, 2, 3, 1)

            predicted = self.audio_encoder(observations)
            target_batch = audio_feature
            return x2, rnn_hidden_states1, predicted, target_batch
        return x2, rnn_hidden_states1

def sample_batch(source, target, audios_feature, batch_size):
    assert len(source) == len(audios_feature), "Audio source and feature shape unmatched"

    source = source.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)
    if batch_size == len(source):
        return source, target, audios_feature    
    indices = torch.randperm(len(source))[:batch_size]
    audios_batch = source[indices]
    feature_batch = audios_feature[indices]
    target_batch = target[indices]
    return audios_batch, target_batch, feature_batch