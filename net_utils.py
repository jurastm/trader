import torch
from torch.distributions import Categorical


def ppo_update(
    model,
    optimizer,
    states,
    actions,
    log_probs_old,
    returns,
    advantages,
    clip_epsilon,
    epochs=4,
    entropy_coef=0.05
):
    """
    states:       (batch_size, seq_len, input_dim) or (batch_size, input_dim)
    actions:      (batch_size,) or (batch_size, seq_len)
    log_probs_old:(batch_size,) or (batch_size, seq_len)
    returns:      (batch_size,) or (batch_size, seq_len)
    advantages:   (batch_size,) or (batch_size, seq_len)
    """
    losses = []

    for _ in range(epochs):
        # Re-run forward pass from scratch (hidden_state=None for each epoch)
        logits, values, _ = model(states, None)
        values = values.squeeze(-1)  # ensure shape matches returns

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - values).pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy
        losses.append(loss.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return torch.mean(torch.stack(losses))


def compute_gae(rewards, masks, values, gamma=0.99, lam=0.95):
    # values is assumed to have one extra element at the end (bootstrap)
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * masks[i] - values[i]
        gae = delta + gamma * lam * masks[i] * gae
        advantages.insert(0, gae)
    return advantages


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