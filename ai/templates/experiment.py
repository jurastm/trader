import sys
import copy
import os
import gym
import numpy as np
import pandas as pd
from gym import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
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


def load_and_preprocess_data(file_path):
    """
    Loads historical Bitcoin price data from a CSV file.
    Expected CSV columns: unix, date, symbol, open, high, low, close, Volume BTC, Volume USD
    """
    df = pd.read_csv(file_path)
    # Convert 'date' to datetime objects
    df['date'] = pd.to_datetime(df['date'])
    # Sort in chronological order
    df = df.sort_values('date').reset_index(drop=True)
    # Forward fill missing values
    # df.fillna(method='ffill', inplace=True)
    df.ffill(inplace=True)
    # Ensure the 'close' column is numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df



class GRUActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions, num_layers=1):
        super(GRUActorCritic, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True  # input shape: (batch_size, seq_len, input_dim)
        )
        
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, h_in=None):
        """
        x: shape (batch_size, seq_len, input_dim)
        h_in: shape (num_layers, batch_size, hidden_dim) or None (zeros if first chunk)
        Returns:
          policy_logits: (batch_size, seq_len, n_actions)
          value:         (batch_size, seq_len, 1)
          h_out:         final hidden state
        """
        # If h_in is None, GRU will default to zeros internally
        out, h_out = self.gru(x, h_in)  # out: (batch_size, seq_len, hidden_dim)
        
        # Flatten out for separate heads
        policy_logits = self.policy_head(out)  # (batch_size, seq_len, n_actions)
        value = self.value_head(out)           # (batch_size, seq_len, 1)
        return policy_logits, value, h_out


class TradingEnv(gym.Env):
    """
    A simplified trading environment for Bitcoin.
    Actions:
        0: Hold
        1: Buy (enter a long position)
        2: Sell (exit a long position)
    Observation: window of shape (window_size, 4)
    Each row => [ (price - sma)/sma, (max_price - sma)/sma, (min_price - sma)/sma, current_position ]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, logger=None, window_size=100, initial_wallet=10000.0):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.logger = logger

        self.window_size = window_size
        self.dataset_size = len(self.df) - self.window_size
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 4), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

        self.initial_wallet = initial_wallet
        self.current_step = 0
        self.position = 0      # 0: flat, 1: long
        self.entry_price = 0.0
        self.wallet = initial_wallet
        self.btc_balance = 0.0

        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.btc_balance = 0.0
        self.entry_price = 0.0
        self.wallet = self.initial_wallet
        return self._next_observation()

    def _next_observation(self):
        """
        Returns a window of size self.window_size, each row has 4 features:
          [ (price - sma)/sma, (max_price - sma)/sma, (min_price - sma)/sma, self.position ]
        """
        start_idx = self.current_step
        end_idx = start_idx + self.window_size
        segment = self.df.loc[start_idx:end_idx - 1]

        obs_window = []
        for i in range(len(segment)):
            row = segment.iloc[i]
            price = row['close']

            window_slice = segment.iloc[max(0, i-5):i+1]  # local range for SMA, etc.
            sma = window_slice['close'].mean()
            min_price = window_slice['low'].min()
            max_price = window_slice['high'].max()
            sma = max(sma, 1e-5)  # avoid division by zero

            f1 = (price - sma) / sma
            f2 = (max_price - sma) / sma
            f3 = (min_price - sma) / sma
            f4 = float(self.position)

            obs_window.append([f1, f2, f3, f4])

        # Pad if segment < window_size:
        if len(obs_window) < self.window_size:
            missing = self.window_size - len(obs_window)
            # Pad with zeros for all 4 features:
            obs_window = [[0.0, 0.0, 0.0, 0.0]] * missing + obs_window

        return np.array(obs_window, dtype=np.float32)

    def _get_current_price(self):
        idx = self.current_step + self.window_size
        if idx >= len(self.df):
            idx = len(self.df) - 1
        return self.df.loc[idx, 'close']

    def step(self, action, penalty=0.0):
        """
        Example step logic:
          - Commission/spread is simplified
          - Reward is scaled profit/loss on sells, small penalty for buying
        """
        current_price = self._get_current_price()
        commission_rate = 1.0 - 0.9975  # 0.0025 => 0.25% commission
        reward = -penalty  # small penalty for each step

        if action == 1:  # Buy
            if self.position == 0 and self.wallet > 0.0:
                cost_before_fee = self.wallet / max(current_price, 1e-5)
                fee = cost_before_fee * commission_rate
                btc_to_buy = cost_before_fee - fee

                # Optional: Negative reward for paying fees:
                reward -= fee

                self.wallet = 0.0
                self.btc_balance = btc_to_buy
                self.entry_price = current_price
                self.position = 1

                if self.logger:
                    msg = (f"[BUY] Step={self.current_step} Price={current_price:.2f} "
                           f"BTC_Bought={btc_to_buy:.6f} Fee={fee:.4f} "
                           f"TotalBalance={self.total_balance:.2f}")
                    self.logger.info(msg)

        elif action == 2:  # Sell
            if self.position == 1 and self.btc_balance > 0.0:
                gross_value = self.btc_balance * current_price
                fee = gross_value * commission_rate
                proceeds = gross_value - fee

                profit = proceeds - (self.btc_balance * self.entry_price)
                # Scale reward, e.g. relative to initial_wallet or total_balance
                reward += profit / max(self.initial_wallet, 1e-5)

                self.wallet += proceeds
                self.btc_balance = 0.0
                self.position = 0
                self.entry_price = 0.0

                if self.logger:
                    msg = (f"[SELL] Step={self.current_step} Price={current_price:.2f} "
                           f"Profit={profit:.4f} Fee={fee:.4f} "
                           f"TotalBalance={self.total_balance:.2f}")
                    self.logger.info(msg)

        # else: Hold => no special logic

        self.current_step += 1
        done = (self.current_step >= self.dataset_size - 1)

        obs = self._next_observation() if not done else np.zeros(
            (self.window_size, 4), dtype=np.float32
        )
        info = {
            "wallet": self.wallet,
            "btc_balance": self.btc_balance,
            "total_balance": self.total_balance,
            "total_profit": self.total_profit
        }
        return obs, reward, done, info

    @property
    def total_balance(self):
        return self.wallet + self.btc_balance * self._get_current_price()

    @property
    def total_profit(self):
        return self.total_balance - self.initial_wallet

    def render(self, mode='human'):
        print(f"Step={self.current_step}, Position={self.position}, "
              f"Wallet={self.wallet:.2f}, BTC={self.btc_balance:.4f}, "
              f"Profit={self.total_profit:.2f}")




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
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return torch.mean(torch.FloatTensor(losses))

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

env = TradingEnv(df, logger=logger, window_size=window_size)

obs_dim = env.observation_space.shape[0]
n_actions = env.action_space.n

print(f'Observation shape: {obs_dim}, Num actions: {n_actions}')

model = GRUActorCritic(window_size=window_size, in_channels=4).to(device)
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