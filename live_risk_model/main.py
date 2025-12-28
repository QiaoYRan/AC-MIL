import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import LiveRoomDataset, collate_fn
from model import LiveRiskModel

def train():
    config = Config()
    dataset = LiveRoomDataset(
        config.data_path, 
        config, 
        desc_vec_path='data/desc_vecs.pt'  # 你的desc向量缓存路径
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn  # 需确保collate_fn返回patch_grid, desc_texts_grid, labels
    )
    model = LiveRiskModel(config).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(config.epochs):
        model.train()
        for patch_grids, desc_texts_grids, labels in dataloader:
            patch_grids = patch_grids.to(config.device)              # [B, U, T, L, 3]
            desc_texts_grids = desc_texts_grids.to(config.device)    # [B, U, T, L, 768]
            labels = labels.to(config.device)                        # [B]
            # Patch Drop（可选）
            # if config.patch_drop_rate > 0:
            #     mask = (torch.rand_like(patch_grids[...,0].float()) > config.patch_drop_rate).unsqueeze(-1).expand_as(patch_grids)
            #     patch_grids = patch_grids * mask
            outputs = model(patch_grids, desc_texts_grids)           # [B, num_classes]
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} finished. Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train() 