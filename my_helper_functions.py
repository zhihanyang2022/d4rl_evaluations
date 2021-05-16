import os
import numpy as np
import gym
import d4rl

def get_dataset_size(dataset_name:str) -> int:
    env = gym.make(dataset_name)
    dataset = env.get_dataset()
    dataset_size = len(dataset['rewards'])
    return dataset_size

def get_log_dir(
        base_dir:str,
        algo_dir:str,
        env_dir:str,
        seed_dir:int
) -> str:
    """Simplifies the progress of making log dirs"""
    return os.path.join(base_dir, algo_dir, env_dir, str(seed_dir))

def get_agent_dir_for_brac(agent_name:str, value_penalty:int) -> str:
    if value_penalty == 0:  # policy regularization only
        agent_dir = f'{agent_name}_pr'
    else:  # both value penalty & policy regularization
        agent_dir = f'{agent_name}_vp'
    return agent_dir