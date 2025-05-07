import copy
import gym
import numpy as np
from gym import spaces


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

    def __init__(self, df, logger, window_size=100, initial_wallet=10000.0):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.logger = logger

        self.num_steps_look_back = window_size
        self.n_steps = len(self.df) - self.num_steps_look_back
        self.dataset_size = len(self.df) - self.num_steps_look_back
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(window_size, 4), dtype=np.float32)

        self.initial_wallet = initial_wallet
        self.num_buy_actions, self.num_sell_actions = 0, 0
        self.position = 0  # 0: no position, 1: long

        self.current_step = 0
        self.btc_balance = 0.0    # Track how many BTC are held

        self.entry_price = 0.0
        self.wallet = self.initial_wallet  # Initialize wallet with deposit

        self.num_resets = 0
        self.reset()

    def reset(self):
        self.current_step = 0
        self.btc_balance = 0.0    # Track how many BTC are held

        self.entry_price = 0.0
        self.wallet = self.initial_wallet  # Initialize wallet with deposit
        cum_rewards = []
        self.num_resets += 1
        return self._next_observation()

    def _next_observation(self):
        window_start = self.current_step
        window_end = window_start + self.num_steps_look_back

        segment = self.df.loc[window_start:window_end-1]

        # Build up the window of features
        obs_window = []
        for i in range(len(segment)):
            row = segment.iloc[i]
            price = row['close']
            
            # Example: compute simple moving average & std from some local context
            sma = segment['close'].iloc[max(0, i-5):i+1].mean()
            min_price = segment['low'].iloc[max(0, i-5):i+1].min()
            max_price = segment['high'].iloc[max(0, i-5):i+1].max()
            
            feature1 = (price - sma) / max(sma, 1)
            feature2 = (max_price - sma) / max(sma, 1)
            feature3 = (min_price - sma) / max(sma, 1)
            feature4 = self.position

            obs_window.append([feature1, feature2, feature3, feature4])

        # If the segment is shorter than window_size (e.g. at the beginning),
        # pad with zeros (or repeat initial values) so we always return (window_size, 3).
        if len(obs_window) < self.num_steps_look_back:
            num_missing = self.num_steps_look_back - len(obs_window)
            obs_window = [[0.0, 0.0, 0.0]] * num_missing + obs_window
        
        return np.array(obs_window, dtype=np.float32)

    
    def _get_current_price(self):
        return self.df.loc[self.current_step + self.num_steps_look_back, 'close']

    def step(self, action, penalty=0.0):
        current_price = self._get_current_price()
        reward = 0.0 - penalty  # Account for the spread
        comission_rate = 1.0 - 0.9975

        if action == 1:  # Buy
            if self.position == 0 and self.wallet > 0.0:

                before_comission = self.wallet / max(current_price, 1e-5)
                fee = before_comission * comission_rate
                btc_to_buy = before_comission - fee

                reward = -comission_rate

                self.wallet = 0.0
                self.btc_balance += btc_to_buy
                self.entry_price = current_price

                self.num_buy_actions += 1
                self.position = 1  # Now holding BTC
                
                if self.logger:
                    str1 = f'Step: {self.current_step}, BTC Bought: {btc_to_buy}, Price: {current_price:.2f}'
                    str2 = f'Balances BTC: {self.btc_balance} USD: {self.wallet} Total: {self.total_balance:.2f}'
                    self.logger.info(str1 + ' ' + str2)
    
        elif action == 2:  # Sell

            if self.position == 1 and self.btc_balance > 0.0:
                before_comission = self.btc_balance * current_price
                fee = before_comission * comission_rate
                proceeds = before_comission - fee

                btc_to_sell = copy.deepcopy(self.btc_balance)
                profit = proceeds - (btc_to_sell * self.entry_price)
                
                reward = profit / self.total_balance
           
                self.wallet += proceeds
                self.btc_balance = 0.0         # All BTC have been sold
                self.entry_price = 0.0

                self.num_sell_actions += 1
                self.position = 0

                if self.logger:
                    str1 = f'Step: {self.current_step}, BTC Sold: {btc_to_sell}, Price: {current_price:.2f}'
                    str2 = f'Balances BTC: {self.btc_balance} USD: {self.wallet} Total: {self.total_balance:.2f}'
                    self.logger.info(str1 + ' ' + str2)

        # else: Hold action does nothing
        self.current_step += 1
        done = self.current_step >= self.dataset_size - 1 # self.n_steps - 1
        obs = self._next_observation() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        info = {
            'usd_balance': self.wallet,
            'btc_balance': self.btc_balance,
            'total_profit': self.total_profit,
            'total_balance': self.total_balance,}
        return obs, reward, done, info
    
    @property
    def total_profit(self):
        return self.wallet + self.btc_balance * self._get_current_price() - self.initial_wallet
    
    @property
    def total_balance(self):
        return self.wallet + self.btc_balance * self._get_current_price()

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, BTC Held: {self.btc_balance}, Wallet: {self.wallet:.2f}, Total Profit: {self.total_profit:.2f}")

