import json
import os
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm

def encode_action_descs(json_path, cache_path, device='cpu'):
    """
    读取 example_data.json，提取所有 action_desc，BERT 编码后存为 {desc: vector} 映射
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    desc_list = []
    for room in data:
        for user in room['user_sequences']:
            for act in user['sequence']:
                desc = act.get('action_desc', '')
                desc_list.append(desc)

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    bert = BertModel.from_pretrained('bert-base-chinese').to(device)
    bert.eval()

    desc_vecs = {}
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(desc_list), batch_size)):
            batch = desc_list[i:i+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = bert(**encoded)
            vecs = out.pooler_output.cpu()  # [B, 768]
            for desc, vec in zip(batch, vecs):
                desc_vecs[desc] = vec.numpy()
    # 保存
    torch.save(desc_vecs, cache_path)
    print(f"Saved desc_vecs to {cache_path}")

def load_desc_vecs(cache_path):
    return torch.load(cache_path) 

