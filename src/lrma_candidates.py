import torch
import numpy as np


def point_to_uniform_candidate_generation(actor_net, state, P=5):
    """
    Point-to-uniform variation method to quantize candidate solutions a_P^t = {a_1(t), ..., a_P(t)}
    (IEEE TNSE 2025 Paper Section V-B.2).
    Generates exactly P candidate solutions sampled from the actor's Categorical distribution.
    
    Args:
        actor_net (nn.Module): ED Actor or Cloud Actor network.
        state (np.ndarray or torch.Tensor): Input state tensor (9-dim for ED, 11-dim for Cloud).
        P (int): Number of candidate solutions (Paper default P=5).
        
    Returns:
        candidates (list of int): List of P sampled discrete action choices.
        probs (torch.Tensor): Softmax probability distribution over actions.
    """
    if isinstance(state, np.ndarray):
        state_t = torch.FloatTensor(state).unsqueeze(0)
    else:
        state_t = state if state.dim() > 1 else state.unsqueeze(0)
        
    with torch.no_grad():
        probs = actor_net(state_t).squeeze(0)
        
    dist = torch.distributions.Categorical(probs)
    candidates = []
    for _ in range(P):
        candidate_act = dist.sample().item()
        candidates.append(candidate_act)
        
    return candidates, probs


def select_best_candidate(candidates, candidate_eval_fn):
    """
    Evaluates candidate action solutions and selects the best candidate via argmax expected reward:
        a*(t) = argmax_{a in a_P^t} r_{expected}(a)
    (IEEE TNSE 2025 Paper Section V-B.2).
    
    Args:
        candidates (list of int): List of P candidate action choices.
        candidate_eval_fn (callable): Evaluation function mapping candidate action -> expected reward / scalar utility.
        
    Returns:
        best_action (int): The candidate action yielding maximum expected reward.
        best_reward (float): The expected reward corresponding to the best candidate action.
    """
    best_action = candidates[0]
    best_reward = float('-inf')
    
    for cand in candidates:
        r_eval = candidate_eval_fn(cand)
        if r_eval > best_reward:
            best_reward = r_eval
            best_action = cand
            
    return best_action, best_reward
