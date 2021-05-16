import numpy as np
from tqdm import tqdm
from collections import deque

GAMMA = 0.99

def compute_mc_returns(rewards):
    mc_return = 0
    mc_returns = deque()
    for r in reversed(rewards):
        mc_return = r + GAMMA * mc_return
        mc_returns.appendleft(mc_return)  # using deque is more efficient lol
    return list(mc_returns)
    
def qlearning_dataset_with_mc_return(env, dataset=None, terminate_on_end=False, **kwargs):
    
    """Made minimal changes (to compute MC returns) from the original qlearning_dataset function from D4RL."""
    
    if dataset is None:
        dataset = env.get_dataset(**kwargs)

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    mc_returns_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    
    rewards_of_current_episode = []
    
    # The author previously used range(N-1), which means that i is up to N-2, the second last index.
    # The problem is that, for the second last transition, both final_timestep and done_bool 
    # are false. As a result, the MC returns for the final episode does not get calculated. 
    #
    # I changed it so that i goes up to N-1, the last index. It turns out that for the last index timeout
    # is always True. This makes final_timestep true and allows MC returns to be calculated.
    
    for i in tqdm(range(N)):  
        
        obs = dataset['observations'][i].astype(np.float32)
        if i + 1 > N - 1:  # N - 1 is the last timestep; so here we are asking, is i+1 an invalid index?
            new_obs = np.zeros_like(obs)
            # Reasoning on why this is the correct thing to do:
            # At the very end, there is only one possible scenario:
            # - final_timestep=True, last transition is ignored (so this full of zeros next state is not used)
        else:
            new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        
        rewards_of_current_episode.append(reward)

        if use_timeouts:  # Always true for our use case.
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        
        # We are using terminate_on_end=False, so the following if statement is entered
        # whenever final_timestep=True. In this case, we ignore the final transition, because
        # the next state is not available, due to the "bad" design of rlkit.
        
        if (not terminate_on_end) and final_timestep:
            
            episode_step = 0
            
            # The last transition is not actually included in the dataset (no next state), but nevertheless 
            # MC returns can consider it with no problem.
            # Essentially, [:-1] deal with the mis-matched length of mc_returns (include last transition) and 
            # other stuff (do not include last transition).
            
            mc_returns = compute_mc_returns(rewards_of_current_episode)[:-1]
            mc_returns_.extend(mc_returns)
            
            rewards_of_current_episode = []
            
            continue  
            
        # If we are here, it means that final_timestep=False. 
        
        # The following if statement is entered if final_timestep=False (otherwise the previous if is entered)
        # and done_bool=True. In this case, we don't put a "continue" at the end because the invalid next state
        # will be ignored during bootstrapping with the help of the done flag. 
        
        # Computing MC returns follows the exact same procedure as in the previous if.
        
        if done_bool or final_timestep:
            
            episode_step = 0
            
            mc_returns = compute_mc_returns(rewards_of_current_episode)
            mc_returns_.extend(mc_returns)
            
            rewards_of_current_episode = []
            
        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
        'mc_returns': np.array(mc_returns_)
    }
