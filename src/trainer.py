from typing import Optional

import torch
from torch.autograd import Variable


class AbstractTrainer:

    def __init__(
        self, model, optimizer, criterion,
        device: Optional[str] = None
    ):
        self.optimizer = optimizer
        self._model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def epoch_train(self, train_loader):
        raise NotImplementedError()

    def epoch_eval(self, eval_loader):
        raise NotImplementedError()

    @property
    def weights(self):
        return self._model.state_dict()


class PolygonRNNTrainer(AbstractTrainer):

    def epoch_train(self, train_loader) -> float:
        dtype = torch.FloatTensor
        dtype_t = torch.LongTensor

        self._model.train()
        epoch_loss: float = 0.0
        for batch_idx, data in enumerate(train_loader):
            x = Variable(data[0].type(dtype)).to(self.device)
            x1 = Variable(data[1].type(dtype)).to(self.device)
            x2 = Variable(data[2].type(dtype)).to(self.device)
            x3 = Variable(data[3].type(dtype)).to(self.device)
            gt = Variable(data[4].type(dtype_t)).to(self.device)

            self.optimizer.zero_grad()

            outputs: torch.Tensor = self._model(x, x1, x2, x3)

            outputs = outputs.contiguous().view(-1, 28 * 28 + 3)
            targets: torch.Tensor = gt.contiguous().view(-1)

            loss = self.criterion(outputs, targets)
            loss.backward()
            epoch_loss += float(loss.item())

            self.optimizer.step()

            output_index: torch.Tensor = torch.argmax(outputs, 1)
            correct = (
                targets == output_index  # type: ignore
            ).type(dtype).sum().item()
            acc = correct * 1.0 / targets.shape[0]

        return epoch_loss / len(train_loader)

    def epoch_eval(self, eval_loader) -> float:
        dtype = torch.FloatTensor
        dtype_t = torch.LongTensor

        self._model.eval()
        epoch_loss: float = 0.0
        for batch_idx, data in enumerate(eval_loader):
            x = Variable(data[0].type(dtype)).to(self.device)
            x1 = Variable(data[1].type(dtype)).to(self.device)
            x2 = Variable(data[2].type(dtype)).to(self.device)
            x3 = Variable(data[3].type(dtype)).to(self.device)
            gt = Variable(data[4].type(dtype_t)).to(self.device)

            outputs: torch.Tensor = self._model(x, x1, x2, x3)

            outputs = outputs.contiguous().view(-1, 28 * 28 + 3)
            targets: torch.Tensor = gt.contiguous().view(-1)

            loss = self.criterion(outputs, targets)

            epoch_loss += float(loss.item())

            output_index: torch.Tensor = torch.argmax(outputs, 1)
            correct = (
                targets == output_index  # type: ignore
            ).type(dtype).sum().item()
            acc = correct * 1.0 / targets.shape[0]

        return epoch_loss / len(eval_loader)

