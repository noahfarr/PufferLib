from functools import lru_cache
from pdb import set_trace as T
from collections import defaultdict

import torch
import copy

import numpy as np
import pandas as pd

from pufferlib.models import Policy

# Provides a pool of policies that collectively process a batch
# of observations. The batch is split across policies according
# to the sample weights provided at initialization.
class PolicyPool():
    def __init__(
        self, learner: Policy,
        batch_size: int,
        num_policies: int,
        learner_weight: float = 1.0):

        self._learner = learner
        self._learner_weight = learner_weight

        self._num_scores = 0
        self._num_policies = num_policies
        self._policies = [learner]

        self._batch_size = batch_size
        self._sample_idxs = self._compute_sample_idxs(batch_size)

        self.learner_mask = np.zeros(batch_size)
        self.learner_mask[self._sample_idxs[0]] = 1
        self.scores = defaultdict(list)

        self._allocated = False

    def _compute_sample_idxs(self, batch_size):
        learner_batch = int(batch_size * self._learner_weight)

        other_batch = 0
        if self._num_policies > 1:
            other_batch = (batch_size - learner_batch) // (self._num_policies - 1)

        # Create indices for splitting data across policies
        sample_weights = [learner_batch] + [other_batch for _ in self._policies]
        print(f"PolicyPool sample_weights: {sample_weights}")
        chunk_size = sum(sample_weights)
        pattern = [i for i, weight in enumerate(sample_weights)
                for _ in range(weight)]

        # Distribute indices among sublists
        sample_idxs = [[] for _ in range(self._num_policies)]
        for idx in range(batch_size):
            sublist_idx = pattern[idx % chunk_size]
            sample_idxs[sublist_idx].append(idx)
        return sample_idxs

    def forwards(self, obs, lstm_state=None, dones=None):
        batch_size = len(obs)
        for samp, policy in zip(self._sample_idxs, self._policies):
            if lstm_state is not None:
                atn, lgprob, _, val, (lstm_state[0][:, samp], lstm_state[1][:, samp]) = policy.get_action_and_value(
                    obs[samp],
                    [lstm_state[0][:, samp], lstm_state[1][:, samp]],
                    dones[samp])
            else:
                atn, lgprob, _, val = policy.get_action_and_value(obs[samp])

            if not self._allocated:
                self.allocated = True

                self.actions = torch.zeros(batch_size, *atn.shape[1:], dtype=int).to(atn.device)
                self.logprobs = torch.zeros(batch_size).to(lgprob.device)
                self.values = torch.zeros(batch_size).to(val.device)

                if lstm_state is not None:
                    self.lstm_h = torch.zeros(self.batch_size, *lstm_state[0].shape[1:]).to(lstm_state[0].device)
                    self.lstm_c = torch.zeros(self.batch_size, *lstm_state[1].shape[1:]).to(lstm_state[1].device)

            self.actions[samp] = atn
            self.logprobs[samp] = lgprob
            self.values[samp] = val.flatten()

            if lstm_state is not None:
                self.lstm_h[samp] = lstm_state[0][:, samp]
                self.lstm_c[samp] = lstm_state[1][:, samp]

        if lstm_state is not None:
            return self.actions, self.logprobs, self.values, (self.lstm_h, self.lstm_c)
        return self.actions, self.logprobs, self.values, None

    def update_scores(self, infos, info_key):
        # TODO: Check that infos is dense and sorted
        agent_infos = []
        for info in infos:
            agent_infos += list(info.values())

        policy_infos = []
        for p in range(len(self._policies)):
            samp = self._sample_idxs[p]
            pol_infos = np.array(agent_infos)[samp]
            policy_infos.append(pol_infos)

            for i in pol_infos:
                if info_key not in i:
                    continue

                self.scores[p].append(i[info_key])
                self.num_scores += 1

        return policy_infos

    # Update the active policies to be used for the next batch. Always
    # include the required policies, and then randomly sample the rest
    # from the available policies.
    def update_policies(self, policies):
        self._policies = [self._learner] + policies