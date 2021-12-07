# Reinforcement Learning with Advantage Actor Critic (A2C)

Training LunarLander in OpenAI Gym using A2C.

## Dependencies
Following packages need to be installed: `numpy`, `matplotlib`, `torch` and `gym`


## Usage
```
$ python a2c_train.py [-h] [--env ENV] [--episodes EPISODES] [--hidden_size HIDDEN_SIZE] [--lr_actor LR_ACTOR]
                    [--lr_critic LR_CRITIC] [--discount DISCOUNT] [--seed SEED] [--outpath OUTPATH] [--save_video]
                    [--progress_step PROGRESS_STEP]

optional arguments:
  -h, --help            show this help message and exit
  --env ENV             Gym environment name (default: LunarLander-v2)
  --episodes EPISODES   Number of episodes to train (default: 400)
  --hidden_size HIDDEN_SIZE
                        Size of hidden layers (default: 32)
  --lr_actor LR_ACTOR   Actor learning rate (default: 0.001)
  --lr_critic LR_CRITIC
                        Critic learning rate (default: 0.001)
  --discount DISCOUNT   Discount factor (default: 0.99)
  --seed SEED           Random seed (default: 42)
  --outpath OUTPATH     Path to save results (default: ./results)
  --save_video          Saves video to outpath (default: False)
  --progress_step PROGRESS_STEP
                        Print progress and save video after every given episodes (default: 10)
```