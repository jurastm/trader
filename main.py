import os
import sys
import torch
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from environment import TradingEnv
from models import ActorCritic, ConvNet
from utils import load_and_preprocess_data
import logging
from tensorboardX import SummaryWriter


logging.basicConfig(
    filename="trading_env.log",
    filemode='a',  # Overwrites on each run; use "a" to append
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

np.set_printoptions(suppress=True)

def ppo_update(model, optimizer, states, actions, log_probs_old, returns, advantages, clip_epsilon, epochs=4, entropy_coef=0.05):
    losses = []
    for _ in range(epochs):
        logits, values = model(states)
        values = values.squeeze(1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - values).pow(2).mean()
        
        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
        losses.append(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return torch.mean(torch.stack(losses))

def compute_returns(rewards, masks, values, gamma):
    """
    Compute discounted returns.
    """
    returns = []
    R = values[-1]
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def compute_gae(rewards, masks, values, gamma=0.99, lam=0.95):
    # values is assumed to have one extra element at the end (bootstrap)
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        advantages.insert(0, gae)
    return advantages


file_path = os.path.expanduser('~/trader/data/BTC-2021min.csv')
df = load_and_preprocess_data(file_path)
print(f'Num data points: {len(df)}')

total_updates = 100000
rollout_length = 1024
gamma = 0.99
clip_epsilon = 0.2
ppo_epochs = 4
lr = 3e-4
window_size = 100
hidden_dim = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

env = TradingEnv(df, logger=logger, window_size=window_size)

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f'Observation shape: {obs_dim}, Num actions: {n_actions}')

model = ConvNet(window_size=window_size, in_channels=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)

# Initialize the writer (set a log directory as needed)
writer = SummaryWriter(log_dir='./logs')

all_rewards, all_actions = [], []
global_step = 0
for update in range(total_updates):
    states, actions, rewards, masks, log_probs, values = [], [], [], [], [], []
    ep_reward = 0

    state = env.reset()
    for step in range(rollout_length):
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0).transpose(1, 2)
        logits, value = model(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, done, info = env.step(action.item())
        ep_reward += reward

        # Save rollout data
        states.append(state)
        actions.append(action.item())
        all_actions.append(action.item())
        rewards.append(reward)
        masks.append(1 - float(done))
        log_probs.append(log_prob.item())
        values.append(value.item())

        state = next_state
        global_step += 1
        
        if done:
            state = env.reset()
            all_rewards.append(ep_reward)
            ep_reward = 0
            
        if (global_step + 1) % env.n_steps == 0:
            all_rewards.append(ep_reward)
            ep_reward = 0
        
        if env.total_balance < 1.0:
            state = env.reset()

    # Convert rollout data to tensors
    states = torch.FloatTensor(states).to(device).transpose(1, 2)
    actions = torch.LongTensor(actions).to(device)
    log_probs_old = torch.FloatTensor(log_probs).to(device)
    values = torch.FloatTensor(values).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    masks = torch.FloatTensor(masks).to(device)

    # Compute returns and advantages
    # Get value of the last state for bootstrapping
    state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0).transpose(1, 2)
    _, next_value = model(state_tensor)
    next_value = next_value.item()
    values = torch.cat((values, torch.FloatTensor([next_value]).to(device)))
    advantages = compute_gae(rewards.cpu().numpy(), masks.cpu().numpy(), values.cpu().numpy(), gamma, lam=0.95)
    advantages = torch.FloatTensor(advantages).to(device)
    returns = compute_returns(rewards.cpu().numpy(), masks.cpu().numpy(), values.cpu().numpy(), gamma)
    returns = torch.FloatTensor(returns).to(device)

    # PPO update
    mean_loss = ppo_update(model, optimizer, states, actions, log_probs_old, returns, advantages, clip_epsilon, ppo_epochs)

    if update % 1 == 0:
        avg_reward = np.mean(all_rewards[-50:]) if all_rewards else 0
        writer.add_scalar('Average_Reward', avg_reward, update)
        writer.add_scalar('Loss', mean_loss.item(), global_step)
        # print(f"Update {update}, Average Reward (last 10 episodes): {avg_reward:.3f}")
        sys.stdout.write(f'\rUpdate {update}, Average Reward (last 50 episodes): {avg_reward:.3f}')
        sys.stdout.flush()

writer.close()

# Save the trained model
torch.save(model.state_dict(), "ppo_trading_bot.pth")
print("Model saved as ppo_trading_bot.pth")

# Plot the rewards
plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Total Reward per Episode")
plt.show()