import torch
import torch.nn as nn

from models import PolygonRNN

batch_size: int = 2
lr: float = 1e-4
dataset_size: int = 4  # Number of training images.

if __name__ == '__main__':
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    net: nn.Module = PolygonRNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[4000, 100000],
        gamma=0.1
    )

    print(net)
