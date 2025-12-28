# %%
from data.data import encode_action_descs
# def encode_action_descs(json_path, cache_path, device='cpu'):
encode_action_descs('data/example_data.json','data/desc_vecs', device='cpu')
# %%
