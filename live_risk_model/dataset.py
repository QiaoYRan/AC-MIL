import json
import torch
from torch.utils.data import Dataset
from config import Config
from utils import pad_or_truncate
from data.data import load_desc_vecs, encode_action_descs

class LiveRoomDataset(Dataset):
    def __init__(self, json_path, config: Config, desc_vec_path=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.config = config
        self.desc_vecs = load_desc_vecs(desc_vec_path) if desc_vec_path else {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        room = self.data[idx]
        # 构建 patch grid: [max_users, max_timeslots, max_patch_seq_len, 3] 
        # 这里只做简单示例，实际应做id映射、embedding等
        user_map = {u['user_id']: i for i, u in enumerate(room['user_sequences'])} 
        patch_grid = []
        for u_idx in range(self.config.max_users):
            if u_idx < len(room['user_sequences']):
                user_seq = room['user_sequences'][u_idx]
                patches = [[] for _ in range(self.config.max_timeslots)]
                for act in user_seq['sequence']:
                    t = int(act['timeslot_5min'])
                    if t < self.config.max_timeslots:
                        patches[t].append([
                            int(act.get('action_name_id', 0)),  # 需预处理映射
                            int(act.get('if_anchor', 0)),
                         #   int(act.get('desc_id', 0)),         # 需预处理映射
                        ])
                # pad每个patch
                patches = [pad_or_truncate(p, self.config.max_patch_seq_len, [0,0]) for p in patches]
            else:
                patches = [ [[0,0]] * self.config.max_patch_seq_len for _ in range(self.config.max_timeslots)]
            patch_grid.append(patches)
        patch_grid = torch.tensor(patch_grid, dtype=torch.long)  # [U, T, L, 2]
        # 假设 patch_grid: [U, T, L, 2]
        # desc_texts_grid: [U, T, L, 768]
        desc_texts_grid = []
        for u_idx in range(self.config.max_users):
            user_descs = []
            if u_idx < len(room['user_sequences']):
                user_seq = room['user_sequences'][u_idx]
                patches = [[] for _ in range(self.config.max_timeslots)]
                for act in user_seq['sequence']:
                    t = int(act['timeslot_5min'])
                    if t < self.config.max_timeslots:
                        desc = act.get('action_desc', '')
                        patches[t].append(self.desc_vecs.get(desc, torch.zeros(768)))
                # pad每个patch
                patches = [patch + [torch.zeros(768)] * (self.config.max_patch_seq_len - len(patch)) if len(patch) < self.config.max_patch_seq_len else patch[:self.config.max_patch_seq_len] for patch in patches]
            else:
                patches = [[torch.zeros(768)] * self.config.max_patch_seq_len for _ in range(self.config.max_timeslots)]
            user_descs.append(patches)
            desc_texts_grid.append(user_descs)
        desc_texts_grid = torch.tensor(desc_texts_grid, dtype=torch.float)  # [U, T, L, 768]
        label = room['label']
        return patch_grid, desc_texts_grid, label

def collate_fn(batch):
    patch_grids, desc_texts_grids, labels = zip(*batch)
    patch_grids = torch.stack(patch_grids)
    desc_texts_grids = torch.stack(desc_texts_grids)
    labels = torch.tensor(labels)
    return patch_grids, desc_texts_grids, labels 