# d4rl_evaluations

This repository contains the algorithms used to evaluate tasks in the [D4RL benchmark](https://github.com/rail-berkeley/d4rl). All code is lightly modified from other public repositories on Github.

## Reference Codebases.

- [AlgaeDICE](https://github.com/google-research/google-research/tree/master/algae_dice)
- [BRAC, BC, BEAR](https://github.com/google-research/google-research/tree/master/behavior_regularized_offline_rl)
- [AWR](https://github.com/xbpeng/awr)
- [BCQ](https://github.com/sfujim/BCQ)
- [Continuous REM](https://github.com/theSparta/off_policy_mujoco)

## By Zhihan

### SSH to Google Cloud Machine

- `ssh-keygen -t rsa -f key_dir -C username`
    - generates key and key.pub
- copy and paste key.pub (open with textEdit) into SSH keys, and save
- connect to instance using private key: `ssh -i $PROJ/offline-rl-notes/key yangz2@35.221.143.120`

### Commands

- Run BCQ: `python bcq/scripts/run_script.py --env_name=halfcheetah-expert-v1`
    - python bcq/scripts/run_script.py --env_name=halfcheetah-random-v1
    - python bcq/scripts/run_script.py --env_name=halfcheetah-medium-v1
    - python bcq/scripts/run_script.py --env_name=halfcheetah-expert-v1
    - python bcq/scripts/run_script.py --env_name=halfcheetah-medium-expert-v1
    - python bcq/scripts/run_script.py --env_name=halfcheetah-medium-replay-v1
    - python bcq/scripts/run_script.py --env_name=halfcheetah-medium-replay-v1; python bcq/scripts/run_script.py --env_name=halfcheetah-random-v1; python bcq/scripts/run_script.py --env_name=halfcheetah-medium-v1; python bcq/scripts/run_script.py --env_name=halfcheetah-medium-expert-v1
- Run BEAR:
    - Default hyperparameters shared by all task (in this codebase):
        - `qf_lr=3e-4`
        - `policy_lr=1e-4`
        - `num_samples=100`
    - These are hyperparameters used for D4RL:
        - Hopper: `python bear/examples/bear_hdf5_d4rl.py --env='hopper-medium-v1' --kernel_type='laplacian' --mmd_sigma=20`
        - Walker2d: `python bear/examples/bear_hdf5_d4rl.py --env='walk2d-medium-v1' --kernel_type='laplacian' --mmd_sigma=20`
        - HalfCheetah: `python bear/examples/bear_hdf5_d4rl.py --env='halfcheetah-medium-v1' --kernel_type='gaussian' --mmd_sigma=20`
            - Test: `python bear/examples/bear_hdf5_d4rl.py --env='halfcheetah-medium-v1' --kernel_type='gaussian' --mmd_sigma=20 --num_trains_per_train_loop=1`

```{bash}
python bear/examples/bear_hdf5_d4rl.py --env='halfcheetah-random-v1' --kernel_type='gaussian' --mmd_sigma=20 && \
python bear/examples/bear_hdf5_d4rl.py --env='halfcheetah-medium-v1' --kernel_type='gaussian' --mmd_sigma=20 && \
python bear/examples/bear_hdf5_d4rl.py --env='halfcheetah-medium-replay-v1' --kernel_type='gaussian' --mmd_sigma=20 && \
python bear/examples/bear_hdf5_d4rl.py --env='halfcheetah-medium-expert-v1' --kernel_type='gaussian' --mmd_sigma=20
```

#### Run CQL(H)

```bash
# working directory should be d4rl_evaluations
python cql/d4rl/examples/cql_mujoco_new.py \
--env=halfcheetah-expert-v1 \
--min_q_weight=5.0 \
--lagrange_thresh=-1.0 \
--policy_lr=1e-4 \
--seed=0 \
--min_q_version=3
```

Run the rest

```
python cql/d4rl/examples/cql_mujoco_new.py --env=halfcheetah-random-v1 --min_q_weight=5.0 --lagrange_thresh=-1.0 --policy_lr=1e-4 --seed=0 --min_q_version=3 && python cql/d4rl/examples/cql_mujoco_new.py --env=halfcheetah-medium-v1 --min_q_weight=5.0 --lagrange_thresh=-1.0 --policy_lr=1e-4 --seed=0 --min_q_version=3 && python cql/d4rl/examples/cql_mujoco_new.py --env=halfcheetah-medium-replay-v1 --min_q_weight=5.0 --lagrange_thresh=-1.0 --policy_lr=1e-4 --seed=0 --min_q_version=3 && python cql/d4rl/examples/cql_mujoco_new.py --env=halfcheetah-medium-expert-v1 --min_q_weight=5.0 --lagrange_thresh=-1.0 --policy_lr=1e-4 --seed=0 --min_q_version=3
```



### Change

#### BCQ

- DEBUG: Added `sys.path.append('bcq/')` in the beginning because `bcq/scripts/run_script.py`
depends on `bcq/continuous_bcq`
- DEBUG: Changed D4RL dataset path into `d4rl.set_dataset_path('/home/yangz2/.d4rl/datasets')`
- DEBUG: changed buffer size to adapt to the size of the offline dataset; previously, buffer size is fixed to 1e6, which
is too small for some datasets (e.g., medium-expert, medium-replay)
- DEBUG: change the default value for `max_timesteps` to 500K grad steps (correspond to D4RL paper)

#### BEAR

- Added `sys.path.append('bear/')` in the beginning because `bcq/examples/bear_hdf5_d4rl.py`
depends on `bear/rlkit`
- Changed `num_epochs` to 100; changed `num_trains_per_train_loop` to 5K; together this is 500K grad steps

### Cautions

- D4RL paper claims to make minimal changes to the existing codebases. This is certainly convenient, but different
algorithms use different hyperparameters (e.g., batch size), which might be problematic for comparison purposes.