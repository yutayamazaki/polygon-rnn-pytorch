import logging.config
import os
from datetime import datetime
from logging import getLogger
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from torchvision import transforms

from datasets import PolygonCityScapesDataset, load_cityscapes
from models import PolygonRNN
from trainer import PolygonRNNTrainer
import utils


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


batch_size: int = 16
lr: float = 1e-4
num_epochs: int = 100
seq_len: int = 60

if __name__ == '__main__':
    utils.seed_everything(seed=428)
    sns.set()

    log_config: Dict[str, Any] = load_yaml('logger_conf.yaml')
    logging.config.dictConfig(log_config)
    logger = getLogger(__name__)

    # Setup directory that saves the experiment results.
    dirname: str = datetime.now().strftime('%Y%m%d_%H-%M-%S')
    save_dir: str = os.path.join('../experiments', dirname)
    os.makedirs(save_dir, exist_ok=False)
    weights_dir: str = os.path.join(save_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=False)

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    net: nn.Module = PolygonRNN()
    net = net.to(device)

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[4000, 100000],
        gamma=0.1
    )

    datasets: Dict[str, List[Tuple[str, str]]] = load_cityscapes()

    train_dataset = PolygonCityScapesDataset(
        city_paths=datasets['train'], transform=transforms.ToTensor()
    )
    valid_dataset = PolygonCityScapesDataset(
        city_paths=datasets['val'], transform=transforms.ToTensor()
    )
    test_dataset = PolygonCityScapesDataset(
        city_paths=datasets['test'], transform=transforms.ToTensor()
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size
    )
    trainer: PolygonRNNTrainer = PolygonRNNTrainer(net, optimizer, criterion)
    best_loss: float = 1e8
    metrics: Dict[str, List[float]] = {'train_loss': [], 'valid_loss': []}
    for epoch in range(1, num_epochs + 1):
        train_loss: float = trainer.epoch_train(train_loader)
        valid_loss: float = trainer.epoch_eval(valid_loader)

        metrics['train_loss'].append(train_loss)
        metrics['valid_loss'].append(valid_loss)

        logger.info(
            f'EPOCH: {str(epoch).zfill(3)}, TRAIN LOSS: {train_loss:.5f}, '
            f'VALID LOSS: {valid_loss:.5f}'
        )

        if valid_loss < best_loss:
            best_loss = valid_loss
            path: str = os.path.join(
                weights_dir,
                f'loss{valid_loss:.5f}_epoch{str(epoch).zfill(3)}.pth'
            )
            torch.save(trainer.weights, path)

    # Plot metrics.
    plt.plot(metrics['train_loss'], label='train')
    plt.plot(metrics['valid_loss'], label='valid')
    plt.title('Loss curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))
    plt.clf()
