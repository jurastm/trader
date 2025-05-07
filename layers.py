import torch.nn as nn

class AdaptiveNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveNormalization, self).__init__()
        # BatchNorm1d tracks running stats during training and uses them at eval time.
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        
    def forward(self, x):
        # x is assumed to have shape (batch_size, num_features)
        return self.bn(x)
    