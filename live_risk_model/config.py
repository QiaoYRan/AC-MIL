class Config:
    data_path = 'data/example_data.json'
    max_users = 32
    max_timeslots = 24
    max_patch_seq_len = 10
    action_vocab_size = 50
    action_emb_dim = 64
    desc_emb_dim = 32
    patch_emb_dim = 128
    num_classes = 2
    batch_size = 8
    lr = 1e-3
    epochs = 10
    device = 'cuda'  # or 'cpu'
    patch_drop_rate = 0.3 