import sys
import gc
import os
import torch
import numpy as np
from torch.distributions import Categorical
from environment2 import TradingEnv
from models import GRUActorCritic
from utils import load_and_preprocess_data
import logging
from tensorboardX import SummaryWriter
import net_utils


logging.basicConfig(
    filename="trading_env.log",
    filemode='a',  # Overwrites on each run; use "a" to append
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger()

np.set_printoptions(suppress=True)

total_updates = 10000
rollout_length = 1024
gamma = 0.99
clip_epsilon = 0.2
ppo_epochs = 4
lr = 3e-4
window_size = 64
hidden_dim = 256
input_dim = 256

file_path = os.path.expanduser('~/trader/data/BTC-2021min.csv')
df = load_and_preprocess_data(file_path)

env = TradingEnv(df, logger=logger, window_size=window_size)
obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
model = GRUActorCritic(input_dim, hidden_dim, n_actions=3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Initialize the writer (set a log directory as needed)
writer = SummaryWriter(log_dir='./logs')

all_rewards, all_actions = [], []
global_step = 0

state = env.reset()
for update in range(total_updates):
    states, actions, rewards, masks, log_probs, values = [], [], [], [], [], []
    
    ep_reward = 0
    hidden_state = None
    for step in range(rollout_length):

        input_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        logits, value, hidden_state = model(input_tensor, hidden_state)
        dist = Categorical(logits=logits.squeeze(1))
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
        
        if done or env.total_balance < 100.0:
            state = env.reset()
            all_rewards.append(ep_reward)
            ep_reward = 0
            hidden_state = None

    all_rewards.append(ep_reward)

    # Convert rollout data to tensors
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    log_probs_old = torch.FloatTensor(log_probs).to(device)
    values = torch.FloatTensor(values).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    masks = torch.FloatTensor(masks).to(device)

    # Compute returns and advantages
    # Get value of the last state for bootstrapping
    state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
    _, next_value, _ = model(state_tensor, hidden_state)
    next_value = next_value.item()
    values = torch.cat((values, torch.FloatTensor([next_value]).to(device)))
    advantages = net_utils.compute_gae(rewards.cpu().numpy(), masks.cpu().numpy(), values.cpu().numpy(), gamma, lam=0.95)
    advantages = torch.FloatTensor(advantages).to(device)
    returns = net_utils.compute_returns(rewards.cpu().numpy(), masks.cpu().numpy(), values.cpu().numpy(), gamma)
    returns = torch.FloatTensor(returns).to(device)

    # PPO update
    mean_loss = net_utils.ppo_update(model, optimizer, states, actions, log_probs_old, returns,
                                     advantages, clip_epsilon, ppo_epochs)

    if update % 1 == 0:
        avg_reward = np.mean(all_rewards[-50:]) if all_rewards else 0
        writer.add_scalar('Episode_Reward', ep_reward, update)
        writer.add_scalar('Avg_Reward', avg_reward, update)
        writer.add_scalar('Loss', mean_loss.item(), update)
        writer.add_scalar('Total_Balance', env.total_balance, update)
        writer.add_scalar('Total_Profit', env.total_profit, update)

        sys.stdout.write(f'\rUpdate {update}, [Avg/Ep] Reward: [{avg_reward:.3f}/{ep_reward:.3f}], Loss: {mean_loss.item():.3f}')
        sys.stdout.flush()

    del mean_loss
    gc.collect()
    torch.cuda.empty_cache()


writer.close()

# Save the trained model
torch.save(model.state_dict(), "ppo_trading_bot.pth")
print("Model saved as ppo_trading_bot.pth")


