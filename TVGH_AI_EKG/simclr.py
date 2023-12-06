import logging
import os
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils import save_checkpoint

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = kwargs['writer']
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        assert similarity_matrix.shape == (
            self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):
        scaler = GradScaler(enabled=self.args.fp16_precision)
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")

        for epoch_counter in range(self.args.epochs+1):
            for images in tqdm(train_loader, ncols=80):
                images = torch.cat(images, dim=0).to(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    # print(self.model(images))
                    # features, _ = self.model(images)
                    features = self.model(images)
                    logits, pseudolabels = self.info_nce_loss(features)
                    loss = self.criterion(logits, pseudolabels)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            self.writer.add_scalar('loss', loss, global_step=epoch_counter)
            self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=epoch_counter)

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            logging.info(f"Epoch: {epoch_counter}\tLoss: {loss:.4f}")

            if epoch_counter % self.args.log_every_n_epochs == 0:
                save_checkpoint({
                    'epoch': epoch_counter,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(self.writer.log_dir, f'checkpoint_{epoch_counter:04d}.pth.tar'))

        logging.info("Training has finished.")
