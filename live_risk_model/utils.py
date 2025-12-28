import torch

def get_position_indices(max_users, max_timeslots, device='cpu'):
    user_idx = torch.arange(max_users, device=device).unsqueeze(1).expand(max_users, max_timeslots)
    time_idx = torch.arange(max_timeslots, device=device).unsqueeze(0).expand(max_users, max_timeslots)
    return user_idx, time_idx

def pad_or_truncate(seq, max_len, pad_value=0):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        return seq + [pad_value] * (max_len - len(seq)) 