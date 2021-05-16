## Run commands

### Run BRAC primal or dual

```{bash}
python brac/scripts/train_brac.py \
--agent_name=brac_primal \
--value_penalty=0 \
--env_name=halfcheetah-medium-v1 \
--seed=0 \
--pretrain=1
```

Arguments:
- `agent_name`: `brac_primal` or `brac_dual`
- `value_penalty`: `0` or `1`
- `env_name`: any task name in D4RL
- `seed`: any integer

### Run BC

```{bash}
python brac/scripts/train_bc.py \
--env_name=halfcheetah-expert-v1 \
--seed=0
```

## Install packages

I've changed `tensorflow-probability=0.8.0rc0` to just `tensorflow-probability` due to compatibility issues.

```{bash}
pip install -r requirements.txt
```

A LOT of packages will be installed and I received an error like this:

```
ERROR: Could not install packages due to an EnvironmentError: [Errno 28] No space left on device
```

It turns out that you can just run `pip install -r requirements.txt` several times to solve this problem because each time you try some additional packages get installed successfully.

Afterwards, you can try to import some packages in your command line Python to verify that their installations were successful.