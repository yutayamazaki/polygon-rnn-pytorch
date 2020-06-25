from logging import getLogger, StreamHandler, INFO

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from datasets import PolygonCityScapesDataset, load_cityscapes
from models import PolygonRNN

logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False

batch_size: int = 2
lr: float = 1e-4
data_size: int = 4  # Number of training images.
num_epochs: int = 10
seq_len: int = 60

dtype = torch.FloatTensor
dtype_t = torch.LongTensor

if __name__ == '__main__':
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    net: nn.Module = PolygonRNN()
    net: nn.Module = net.to(device)

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[4000, 100000],
        gamma=0.1
    )

    datasets = load_cityscapes()

    dtrain = PolygonCityScapesDataset(city_paths=datasets['train'], transform=transforms.ToTensor())
    dval = PolygonCityScapesDataset(city_paths=datasets['val'], transform=transforms.ToTensor())
    dtest = PolygonCityScapesDataset(city_paths=datasets['test'], transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        dtrain, batch_size=batch_size,
        shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dtrain, batch_size=batch_size)

    for epoch in range(1, num_epochs + 1):
        for batch_idx, data in enumerate(train_loader):
            x = Variable(data[0].type(dtype))
            x1 = Variable(data[1].type(dtype))
            x2 = Variable(data[2].type(dtype))
            x3 = Variable(data[3].type(dtype))
            gt = Variable(data[4].type(dtype_t))

            optimizer.zero_grad()

            outputs: torch.Tensor = net(x, x1, x2, x3)

            outputs: torch.Tensor = outputs.contiguous().view(-1, 28 * 28 + 3)
            targets: torch.Tensor = gt.contiguous().view(-1)

            loss = criterion(outputs, targets)
            loss.backward()

            output_index: torch.Tensor = torch.argmax(outputs, 1)
            correct: float = (targets == output_index).type(dtype).sum().item()
            acc: float = correct * 1.0 / targets.shape[0]

            msg: str = f'EPOCH: {epoch}/{num_epochs}, ' + \
                       f'BATCH: {batch_idx + 1}, ACC: {acc}'
            logger.info(msg)

            optimizer.step()
            scheduler.step()
