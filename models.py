import torch
import torch.nn as nn
from layers import AdaptiveNormalization


import torch
import torch.nn as nn


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


class ConvNet(nn.Module):
    def __init__(self, window_size, in_channels=3, out_channels=16, kernel_size=3, hidden_dim=64, n_actions=3):
        super(ConvNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(out_channels * (window_size - kernel_size + 1), hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, in_channels, window_size)
        out = self.conv1(x)  # -> (batch_size, out_channels, new_length)
        out = self.relu(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.fc(out)
        out = self.relu(out)

        policy_logits = self.policy_head(out)
        value = self.value_head(out)
        return policy_logits, value


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 64),
            AdaptiveNormalization(64),
            nn.ReLU(),
            nn.Linear(64, 64),
            AdaptiveNormalization(64),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(64, n_actions)
        self.value_head = nn.Linear(64, 1)
        
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value
