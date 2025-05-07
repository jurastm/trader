import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

# -----------------------------
# Custom Trading Environment
# -----------------------------
class TradingEnv(gym.Env):
    """
    A simple trading environment for Bitcoin using historical data.
    Actions:
        0: Hold
        1: Buy (enter a long position)
        2: Sell (exit a long position)
    Observation:
        [current_price, simple_moving_average, current_position]
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.n_steps = len(self.df)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0  # 0: no position, 1: long
        self.entry_price = 0.0
        self.total_profit = 0.0
        return self._next_observation()

    def _next_observation(self):
        current_price = self.df.loc[self.current_step, 'close']
        start = max(0, self.current_step - 4)
        sma = self.df.loc[start:self.current_step, 'close'].mean()
        return np.array([current_price, sma, self.position], dtype=np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        reward = 0.0

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
        elif action == 2:  # Sell
            if self.position == 1:
                profit = current_price - self.entry_price
                reward = profit
                self.total_profit += profit
                self.position = 0
                self.entry_price = 0.0
        # else: Hold action does nothing

        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {"total_profit": self.total_profit}
        return obs, reward, done, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Position: {self.position}, Total Profit: {self.total_profit}")

# -----------------------------
# Data Preprocessing Function
# -----------------------------
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
    df.fillna(method='ffill', inplace=True)
    # Ensure the 'close' column is numeric
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    return df

# -----------------------------
# Actor-Critic Network for PPO
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)

    def forward(self, x):
        x = self.fc(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

# -----------------------------
# Helper Functions for PPO
# -----------------------------
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

def ppo_update(model, optimizer, states, actions, log_probs_old, returns, advantages, clip_epsilon, epochs=4):
    for _ in range(epochs):
        # Get current policy output
        logits, values = model(states)
        values = values.squeeze(1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        
        # Ratio for clipping
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        
        # PPO loss components
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - values).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss  # You can add an entropy bonus if desired
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# -----------------------------
# Main Training Routine
# -----------------------------
def main():
    # Hyperparameters
    file_path = "bitcoin_price.csv"  # Adjust path as needed
    total_updates = 1000
    rollout_length = 128
    gamma = 0.99
    clip_epsilon = 0.2
    ppo_epochs = 4
    lr = 3e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data and create environment
    df = load_and_preprocess_data(file_path)
    env = TradingEnv(df)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize the actor-critic model and optimizer
    model = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    all_rewards = []
    
    for update in range(total_updates):
        states = []
        actions = []
        rewards = []
        masks = []
        log_probs = []
        values = []

        state = env.reset()
        ep_reward = 0

        for step in range(rollout_length):
            state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
            logits, value = model(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample() 
            log_prob = dist.log_prob(action)
            
            next_state, reward, done, info = env.step(action.item())
            ep_reward += reward
            
            # Save rollout data
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            masks.append(1 - float(done))
            log_probs.append(log_prob.item())
            values.append(value.item())

            state = next_state
            if done:
                state = env.reset()
                all_rewards.append(ep_reward)
                ep_reward = 0

        # Convert rollout data to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        log_probs_old = torch.FloatTensor(log_probs).to(device)
        values = torch.FloatTensor(values).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        masks = torch.FloatTensor(masks).to(device)

        # Compute returns and advantages
        # Get value of the last state for bootstrapping
        state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
        _, next_value = model(state_tensor)
        next_value = next_value.item()
        values = torch.cat((values, torch.FloatTensor([next_value]).to(device)))
        returns = compute_returns(rewards.cpu().numpy(), masks.cpu().numpy(), values.cpu().numpy(), gamma)
        returns = torch.FloatTensor(returns).to(device)
        advantages = returns - values[:-1]
        
        # PPO update
        ppo_update(model, optimizer, states, actions, log_probs_old, returns, advantages, clip_epsilon, ppo_epochs)
        
        if update % 50 == 0:
            avg_reward = np.mean(all_rewards[-10:]) if all_rewards else 0
            print(f"Update {update}, Average Reward (last 10 episodes): {avg_reward:.3f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "ppo_trading_bot.pth")
    print("Model saved as ppo_trading_bot.pth")
    
    # Plot the rewards
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward per Episode")
    plt.show()

if __name__ == "__main__":
    main()
