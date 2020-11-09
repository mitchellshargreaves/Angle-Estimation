import torch
from torch import optim
from tqdm.notebook import tqdm

from math import pi

from ML.metrics import *


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

# Function for the training
def train(model, train_loader, loss_fn, optimizer, device):
    model.train() # puts the model in training mode
    running_loss = 0
    with tqdm(total=len(train_loader)) as pbar:
        for (x1, x2), labels in iter(train_loader): # loops through training data
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device) # puts the data on the GPU

#             forward + backward + optimize
            model.hidden = repackage_hidden(model.hidden)
#             model.hiddens = [repackage_hidden(x) for x in model.hiddens]
            optimizer.zero_grad() # clear the gradients in model parameters
            outputs = model((x1, x2)) # forward pass and get predictions
            loss = loss_fn(outputs.squeeze(1), labels) # calculate loss
            loss.backward() # calculates gradient w.r.t to loss for all parameters in model that have requires_grad=True
            optimizer.step() # iterate over all parameters in the model with requires_grad=True and update their weights.

            running_loss += loss.item() # sum total loss in current epoch for print later

            pbar.update(1) #increment our progress bar

    return running_loss/len(train_loader) # returns the total training loss for the epoch

# Function for the validation pass
def validation(model, val_loader, loss_fns, device):
    model.eval() # puts the model in validation mode
    running_loss = [0] * len(loss_fns)
    true_fn = TrueAngLoss(360)
    first = True

    with torch.no_grad(): # save memory by not saving gradients which we don't need
        with tqdm(total=len(val_loader)) as pbar:
            for (x1, x2), labels in iter(val_loader):
                x1, x2, labels = x1.to(device), x2.to(device), labels.to(device) # puts the data on the GPU
                outputs = model((x1, x2)) # passes image to the model, and gets a ouput which is the class probability prediction

                for i, loss_fn in enumerate(loss_fns):
                    val_loss = loss_fn(outputs.squeeze(1), labels) # calculates val_loss from model predictions and true labels
                    running_loss[i] += val_loss.item()

                if not first:
                    running_true += true_fn(outputs.squeeze(1), labels)
                else:
                    running_true = true_fn(outputs.squeeze(1), labels)
                    first = False

                pbar.update(1)

        return [x / len(val_loader) for x in running_loss], running_true / len(val_loader) # return loss value, accuracy

class SensorTrainer():
    def __init__(self, model, dataset, device="cpu", split=.7, bs=64):
        self.device = device

        model.to(device)
        self.model = model
        self.training_loader, self.validation_loader = self._make_loaders(dataset, bs, split)

        self.loss_fn = L2AngLoss(2) # -1 -> 1
        self.metrics = [L1AngLoss(2), L2AngLoss(2)]
        self.optimizer = optim.Adam(model.parameters(), lr=8e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, .95, last_epoch=-1)

    def _make_loaders(self, dataset, bs, split):
        split_idx = round(split * len(dataset))
        split_idx = split_idx - (split_idx % bs)

        train_ds, valid_ds = torch.utils.data.random_split(dataset, (split_idx, len(dataset) - split_idx))

        training_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(valid_ds, batch_size=bs, shuffle=False)

        return training_loader, validation_loader

    def __call__(self, epochs):
        trains = []
        l1s = []
        l2s = []
        breakdowns = []


        for epoch in range(epochs): # loops through number of epochs
            train_loss = train(self.model, self.training_loader, self.loss_fn, self.optimizer, self.device)  # train the model for one epoch
            losses, true_loss = validation(self.model, self.validation_loader, self.metrics, self.device) # after training for one epoch, run the validation() function to see how the model is doing on the validation dataset
            l1_loss, l2_loss = losses
            breakdown = true_loss.cpu().numpy()

            breakdowns.append(breakdown)
            trains.append(train_loss)
            l1s.append(l1_loss)
            l2s.append(l2_loss)

            if epoch > 10:
                self.scheduler.step()

            print("Epoch: {}/{}, Training Loss: {:.6f}, L2 Loss: {:.6f}, L1 Loss: {:.6f},\nAng Err Degrees: {:.4f}, Ang Err Rads: {:.4f}".format(epoch + 1, epochs, train_loss, l2_loss, l1_loss, l1_loss * 180, l1_loss * pi))
            print("Breakdown:", breakdown)
            print('-' * 20)

        print("Finished Training")
        return trains, l1s, l2s, breakdowns