import copy
import gym
import numpy as np
from gym import spaces
import utils

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
        self.commission_rate = 1.0 - 0.9975  # 0.0025 => 0.25% commission
        self.scaling_factor = 100.0

        self.reset()

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.btc_balance = 0.0
        self.entry_price = 0.0
        self.wallet = self.initial_wallet
        return self._next_observation()
    
    def _fft_features(self, input_array, use_log_returns=True, w_size=32):
        # normalized_by_mean = input_array - np.mean(input_array)
        magnitudes, phases, fft_result = utils.fft(input_array,
                                                   use_log_returns=use_log_returns,
                                                   window_size=w_size)
        return np.concatenate((phases, magnitudes[1:], np.array([input_array[-1]])))

    def _next_observation(self):
        """
        Returns a window of size self.window_size, each row has 4 features:
          [ (price - sma)/sma, (max_price - sma)/sma, (min_price - sma)/sma, self.position ]
        """
        start_idx = self.current_step
        end_idx = start_idx + self.window_size
        segment = self.df.loc[start_idx:end_idx - 1]

        close_price = segment['close'].__array__()
        low_price = segment['low'].__array__()
        high_price = segment['high'].__array__()
        volume = segment['Volume BTC'].__array__()

        close_price_features = self._fft_features(close_price, w_size=32)
        low_price_features = self._fft_features(low_price, w_size=32)
        high_price_features = self._fft_features(high_price, w_size=32)
        volume_features = self._fft_features(volume, w_size=32)

        state = np.concatenate(
            (close_price_features, low_price_features, high_price_features, volume_features),
            axis=0)

        return state.astype(np.float32)
        # obs_window = []

        # for i in range(len(segment)):
        #     row = segment.iloc[i]
        #     price = row['close']

        #     sma = row
        #     # window_slice = segment.iloc[max(0, i-5):i+1]  # local range for SMA, etc.
        #     # sma = window_slice['close'].mean()
        #     # min_price = window_slice['low'].min()
        #     # max_price = window_slice['high'].max()
        #     # sma = max(sma, 1e-5)  # avoid division by zero

        #     f1 = (price - sma) / sma
        #     f2 = (max_price - sma) / sma
        #     f3 = (min_price - sma) / sma
        #     f4 = float(self.position)

        #     obs_window.append([f1, f2, f3, f4])

        # # Pad if segment < window_size:
        # if len(obs_window) < self.window_size:
        #     missing = self.window_size - len(obs_window)
        #     # Pad with zeros for all 4 features:
        #     obs_window = [[0.0, 0.0, 0.0, 0.0]] * missing + obs_window

        # return np.array(obs_window, dtype=np.float32)

    def _get_current_price(self):
        idx = self.current_step + self.window_size
        if idx >= len(self.df):
            idx = len(self.df) - 1
        return self.df.loc[idx, 'close']
    
    def log_step(self, fee, btc_to_buy=None, profit=None):
        if btc_to_buy is not None:
            msg = (f"[BUY] Step={self.current_step} Price={self._get_current_price():.2f} "
                f"BTC_Bought={btc_to_buy:.6f} Fee={fee:.4f} "
                f"TotalBalance={self.total_balance:.2f}")
        elif profit is not None:
            msg = (f"[SELL] Step={self.current_step} Price={self._get_current_price():.2f} "
                f"Profit={profit:.4f} Fee={fee:.4f} "
                f"TotalBalance={self.total_balance:.2f}")
        if self.logger:
            self.logger.info(msg)


    def step(self, action, penalty=1e-7):
        """
        Example step logic:
          - Commission/spread is simplified
          - Reward is scaled profit/loss on sells, small penalty for buying
        """
        current_price = self._get_current_price()
        
        reward = -penalty  # small penalty for each step

        fee = 0.0
        if action == 1:  # Buy
            if self.position == 0 and self.wallet > 0.0:
                fee = self.wallet * self.commission_rate
                btc_to_buy = (self.wallet - fee) / max(current_price, 1e-5)
                reward -= self.commission_rate

                self.wallet = 0.0
                self.btc_balance, self.entry_price, self.position = btc_to_buy, current_price, 1

                self.log_step(fee, btc_to_buy)

        elif action == 2:  # Sell
            if self.position == 1 and self.btc_balance > 0.0:
                gross_value = self.btc_balance * current_price
                fee = gross_value * self.commission_rate
                proceeds = gross_value - fee

                profit = proceeds - (self.btc_balance * self.entry_price)
                reward += profit / max(self.total_balance, 1e-5)

                self.wallet += proceeds
                self.btc_balance, self.entry_price, self.position = 0.0, 0.0, 0

                self.log_step(fee, profit=profit)

        reward *= self.scaling_factor
        
        self.current_step += 1
        done = (self.current_step >= self.dataset_size - 1)

        obs = self._next_observation() if not done else np.zeros(
            (self.window_size, 4), dtype=np.float32
        )
        info = {
            "wallet": self.wallet,
            "btc_balance": self.btc_balance,
            'fee': fee,
            'action': action,
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
