import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.optim import Adam
from models import Actor, Critic
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import argparse
from pathlib import Path


def train_a2c_one_ep(actor, critic, optim_actor, optim_critic, env, args):
    total_reward = 0
    ep_len = 0
    done = False
    state = env.reset()

    while not done:
        policy = actor(state)
        action = policy.sample()
        if args.recorder:
            args.recorder.capture_frame()
        next_state, reward, done, _ = env.step(action.detach().cpu().numpy())
        advantage = (reward + (1 - done) * args.discount * critic(next_state) - critic(state))

        total_reward += reward
        state = next_state

        optim_critic.zero_grad()
        critic_loss = advantage**2
        critic_loss.backward()
        optim_critic.step()

        optim_actor.zero_grad()
        actor_loss = -policy.log_prob(action)*advantage.detach()
        actor_loss.backward()
        optim_actor.step()

        ep_len += 1

    if args.recorder:
        args.recorder.close()
    return actor_loss.item(), critic_loss.item(), total_reward, ep_len


def plot_results(data_array):
    plt.plot(data_array[:,2], label='total_rewards')
    filter_size = 5
    smoothed_rewards = np.convolve(data_array[:,2], np.ones(filter_size)/filter_size, mode='same')
    plt.plot(smoothed_rewards, label='total_rewards_smoothed')
    plt.legend()
    plt.show()
    plt.plot(data_array[:,0], label='actor_loss')
    plt.legend()
    plt.show()
    plt.plot(data_array[:,1], label='critic_loss')
    plt.legend()
    plt.show()
    plt.plot(data_array[:,3], label='ep_len')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', type=str, default='LunarLander-v2', help='Gym environment name')
    parser.add_argument('--episodes', type=int, default=400, help='Number of episodes to train')
    parser.add_argument('--hidden_size', type=int, default=32, help='Size of hidden layers')
    parser.add_argument('--lr_actor', type=float, default=1e-3, help='Actor learning rate')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic learning rate')
    parser.add_argument('--discount', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--outpath', type=str, default='./results', help='Path to save results')
    parser.add_argument('--save_video', action='store_true', help='Saves video to outpath')
    parser.add_argument('--progress_step', type=int, default=10, help='Print progress and save video after every given episodes')
   
    args = parser.parse_args()

    outpath = Path(args.outpath)
    outpath.mkdir(parents=True, exist_ok=True)
    
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    n = env.action_space.n

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    env.seed(args.seed)

    actor = Actor(obs_dim, args.hidden_size, n, device).to(device)
    critic = Critic(obs_dim, args.hidden_size, 1, device).to(device)

    optim_actor = Adam(actor.parameters(), lr=args.lr_actor)
    optim_critic = Adam(critic.parameters(), lr=args.lr_critic)

    print('Training')
    data_array = []
    for i in range(args.episodes):
        args.recorder = VideoRecorder(env, f'{outpath}/{args.env}_{i}.mp4') if i % args.progress_step == 0 and args.save_video else None
        actor_loss, critic_loss, total_rewards, ep_len = train_a2c_one_ep(actor, critic, optim_actor, optim_critic, env, args)
        data_array.append([actor_loss, critic_loss, total_rewards, ep_len])
        if i % args.progress_step == 0:
            print(f'episode: [{i+1}/{args.episodes}] '
                f'{actor_loss=:.6f} '
                f'{critic_loss=:.6f} '
                f'{total_rewards=:.2f} '
                f'{ep_len=}')
        
    data_array = np.array(data_array)
    np.savetxt(f'{outpath}/{args.env}_{args.episodes}_{args.seed}.txt', data_array)
    plot_results(data_array)
