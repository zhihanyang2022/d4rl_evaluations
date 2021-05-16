# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline training binary."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('brac')
sys.path.append('.')
import my_helper_functions as mhf  # added/modified by Zhihan

import os

from absl import app
from absl import flags
from absl import logging


import gin
import tensorflow as tf0
import tensorflow.compat.v1 as tf

from behavior_regularized_offline_rl.brac import agents
from behavior_regularized_offline_rl.brac import train_eval_offline
from behavior_regularized_offline_rl.brac import utils

import d4rl

tf0.compat.v1.enable_v2_behavior()

# =========== parse arguments ==========
# All irrelevant arguments have been commented out.

# agent info
# flags.DEFINE_string('agent_name', 'brac_primal', 'agent name.')
# flags.DEFINE_integer('value_penalty', 0, '')
# flags.DEFINE_float('alpha', 1.0, '')

# env info
flags.DEFINE_string('env_name', 'halfcheetah-random-v0', 'env name.')
flags.DEFINE_integer('seed', 0, 'random seed, mainly for training samples.')

# training info
flags.DEFINE_integer('total_train_steps', int(5e5), '')  # 500K grad steps; eval_freq is default 5000
flags.DEFINE_integer('n_eval_episodes', 10, '')

# logging info
flags.DEFINE_string('root_dir', 'results', '')  # added/modified by Zhihan

flags.DEFINE_multi_string('gin_file', None, 'Paths to the gin-config files.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin binding parameters.')

# flags.DEFINE_string('sub_dir', 'auto', '')
# flags.DEFINE_integer('n_train', int(1e6), '')
# flags.DEFINE_integer('save_freq', 1000, '')

FLAGS = flags.FLAGS

# =====================================

def main(_):
  # Setup log dir.
  # if FLAGS.sub_dir == 'auto':
  #   sub_dir = utils.get_datetime()
  # else:
  #   sub_dir = FLAGS.sub_dir

  log_dir = os.path.join(
      FLAGS.root_dir,
      'BC',
      FLAGS.env_name,
      str(FLAGS.seed),
  )  # simplified; added/modified by Zhihan

  model_arch = ((200,200),)
  opt_params = (('adam', 5e-4),)
  utils.maybe_makedirs(log_dir)
  train_eval_offline.train_eval_offline(
      log_dir=log_dir,
      data_file=None,
      agent_module=agents.AGENT_MODULES_DICT['bc'],
      env_name=FLAGS.env_name,
      n_train=mhf.get_dataset_size(FLAGS.env_name),  # added/modified by Zhihan
      total_train_steps=FLAGS.total_train_steps,
      n_eval_episodes=FLAGS.n_eval_episodes,  # added/modified by Zhihan
      model_params=model_arch,
      optimizers=opt_params,
      save_freq=FLAGS.total_train_steps + 100  # I don't want any models to be saved; added/modified by Zhihan
  )

if __name__ == '__main__':
  app.run(main)
