import sys

sys.path.append('.')
import my_helper_functions as mhf
from qlearning_dataset_with_mc_return import qlearning_dataset_with_mc_return
from qlearning_dataset_with_next_action import qlearning_dataset_wonjoon, qlearning_dataset_with_next_action

sys.path.append('cql/d4rl')

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.env_replay_buffer_with_return import EnvReplayBufferWithReturn
from rlkit.data_management.env_replay_buffer_with_next_action import EnvReplayBufferWithNextAction
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.launchers.launcher_util import setup_logger
from rlkit.samplers.data_collector import MdpPathCollector, CustomMDPPathCollector
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.sac.cql import CQLTrainer
from rlkit.torch.networks import FlattenMlp
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm

import argparse, os
import numpy as np

import h5py
import d4rl, gym

import shutil


def load_hdf5(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)  
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


def load_hdf5_with_mc_return(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)  
    replay_buffer._mc_returns = np.expand_dims(np.squeeze(dataset['mc_returns']), 1)
    replay_buffer._size = dataset['terminals'].shape[0]
    print ('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size


def load_hdf5_with_next_action(dataset, replay_buffer):
    replay_buffer._observations = dataset['observations']
    replay_buffer._next_obs = dataset['next_observations']
    replay_buffer._actions = dataset['actions']
    replay_buffer._rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
    replay_buffer._terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
    replay_buffer._next_actions = dataset['next_actions']
    replay_buffer._size = dataset['terminals'].shape[0]
    print('Number of terminals on: ', replay_buffer._terminals.sum())
    replay_buffer._top = replay_buffer._size

    
def experiment(variant):
    eval_env = gym.make(variant['env_name'])
    expl_env = eval_env
    
    obs_dim = expl_env.observation_space.low.size
    action_dim = eval_env.action_space.low.size

    M = variant['layer_size']
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M], 
    )
    eval_policy = MakeDeterministic(policy)
    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )
    expl_path_collector = CustomMDPPathCollector(
        eval_env,
    )
    buffer_filename = None
    if variant['buffer_filename'] is not None:
        buffer_filename = variant['buffer_filename']

    # =========================================================
    # different dataset modifications
    
    if variant['use_sil']:

        print('Internal report: loading data for CQL SIL')

        replay_buffer = EnvReplayBufferWithReturn(
            variant['replay_buffer_size'],
            expl_env,
        )
        
        load_hdf5_with_mc_return(qlearning_dataset_with_mc_return(eval_env), replay_buffer)

    elif variant['cql_beta']:

        print('Internal report: Loading data for CQL beta')

        replay_buffer = EnvReplayBufferWithNextAction(
            variant['replay_buffer_size'],
            expl_env,
        )

        load_hdf5_with_next_action(qlearning_dataset_with_next_action(variant['env_name']), replay_buffer)
        
    else:  # do the standard thing

        replay_buffer = EnvReplayBuffer(
            variant['replay_buffer_size'],
            expl_env,
        )
        
        # if variant['load_buffer'] and buffer_filename is not None:
        #     replay_buffer.load_buffer(buffer_filename)
        # elif 'random-expert' in variant['env_name']:
        #     load_hdf5(d4rl.basic_dataset(eval_env), replay_buffer)
        # else:
        #     load_hdf5(d4rl.qlearning_dataset(eval_env), replay_buffer)
        load_hdf5(qlearning_dataset_wonjoon(variant['env_name']), replay_buffer)

    # =========================================================
       
    trainer = CQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **variant['trainer_kwargs']
    )
    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        eval_both=False,  # added/modified by Zhihan
        batch_rl=variant['load_buffer'],
        **variant['algorithm_kwargs']
    )
    algorithm.to(ptu.device)
    algorithm.train()

def enable_gpus(gpu_str):
    if (gpu_str is not ""):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    return

if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        layer_size=256,
        replay_buffer_size=int(2E6),
        buffer_filename=None,
        load_buffer=None,
        env_name='Hopper-v2',
        sparse_reward=False,
        use_sil=False,  # added for the new SIL idea; default to be false
        algorithm_kwargs=dict(
            num_epochs=3000,
            num_eval_steps_per_epoch=1000,
            num_trains_per_train_loop=1000,  
            num_expl_steps_per_train_loop=1000,
            min_num_steps_before_training=1000,
            max_path_length=1000,
            batch_size=256,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=40000,
            num_qs=2,

            # min Q
            temp=1.0,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=True,   # Defaults to true
            lagrange_thresh=10.0,
            
            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
        ),
    )

    # added/modified by Zhihan

    # Arguments that should be specified
    # env

    # According to instructions in the codebase, for Gym Mujoco tasks, we should use:
    # min_q_weight: 5.0 (different from default)
    # lagrange_thresh: -1.0 (different from default)
    # policy_lr: 1e-4

    # Variants:
    # min_q_version: 3 (CQL(H) default) vs 2 (CQL(rho))

    # For convenience, we should use
    # seed: 10 -> 0 (different from default)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default='hopper-medium-v0')
    parser.add_argument("--gpu", default='0', type=str)
    parser.add_argument("--max_q_backup", type=str, default="False")          # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--deterministic_backup", type=str, default="True")   # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--policy_eval_start", default=10000, type=int)       # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument('--min_q_weight', default=1.0, type=float)            # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--policy_lr', default=1e-4, type=float)              # Policy learning rate
    parser.add_argument('--min_q_version', default=3, type=int)               # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho)) 
    parser.add_argument('--lagrange_thresh', default=5.0, type=float)         # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--use_sil', default='False', type=str)  # added for the new idea
    parser.add_argument('--cql_beta', default='False', type=str)  # added for the new idea

    args = parser.parse_args()
    enable_gpus(args.gpu)
    
    variant['use_sil'] = (True if args.use_sil == 'True' else False)
    variant['cql_beta'] = (True if args.cql_beta == 'True' else False)

    assert not (variant['use_sil'] and variant['cql_beta']), "can't use these two together at this point"
    
    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    if args.lagrange_thresh < 0.0:
        variant['trainer_kwargs']['with_lagrange'] = False
    
    variant['buffer_filename'] = None

    variant['load_buffer'] = True
    variant['env_name'] = args.env
    variant['seed'] = args.seed

    # added/modified by Zhihan: use entire buffer
    variant['replay_buffer_size'] = mhf.get_dataset_size(args.env)

    # added/modified by Zhihan: use 1M grad steps, report avg return across 10 episodes per 10K grad steps
    variant['algorithm_kwargs']['num_epochs'] = 100
    variant['algorithm_kwargs']['num_trains_per_train_loop'] = int(1e4)
    variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = 10 * variant['algorithm_kwargs']['max_path_length']

    print('Epochs:', variant['algorithm_kwargs']['num_epochs'])
    print('Num trains per epoch:', variant['algorithm_kwargs']['num_trains_per_train_loop'])

    # added/modified by Zhihan: convenient log dir

    algo_name = 'CQL'
    if variant['use_sil']:
        algo_name += '_SIL'
    elif variant['cql_beta']:
        algo_name += '_BETA'

    log_dir = mhf.get_log_dir(
            base_dir='results',
            algo_dir=algo_name,
            env_dir=args.env,
            seed_dir=args.seed
    )

    print('Log dir:', log_dir)

    shutil.rmtree(log_dir)  # overwrite any previous stuff written in here by deleting the directory
    # later on setup_logger would re-create it anyway

    setup_logger(
        log_dir=log_dir
    )

    ptu.set_gpu_mode(True)
    experiment(variant)
