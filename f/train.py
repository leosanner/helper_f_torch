import torch
from torch import nn
import numpy as np
from tqdm.auto import tqdm
from timeit import default_timer as timer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def acc_fn(y_true, y_pred):
  return (torch.eq(y_true, y_pred).sum().item()/ len(y_pred)) * 100


def train_step(dl: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = device):

  train_loss, acc = 0, 0
  model.to(device)

  for batch, (X, y) in enumerate(dl):
    X, y = X.to(device), y.to(device)

    logit = model(X)
    pred = torch.softmax(logit, dim=1).argmax(dim=1)
    loss = loss_fn(logit, y)
    acc_ = acc_fn(y, pred)

    train_loss += loss
    acc += acc_

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

  train_loss /= len(dl)
  acc /= len(dl)

  return (train_loss, acc)


def test_step(dl: torch.utils.data.DataLoader,
               model: torch.nn.Module,
               loss_fn: torch.nn.Module,
               device: torch.device = device):

  test_loss, acc = 0, 0
  model.to(device)

  with torch.inference_mode():
    model.eval()

    for X, y in dl:
      X, y = X.to(device), y.to(device)
      logit = model(X)
      pred = torch.softmax(logit, dim=1).argmax(dim=1)

      loss = loss_fn(logit, y)
      acc_ = acc_fn(y, pred)

      test_loss += loss
      acc += acc_

    acc /= len(dl)
    test_loss /= len(dl)

  return (test_loss, acc)


def model_fit(dl_train: torch.utils.data.DataLoader,
              dl_test: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              optimizer: torch.optim.Optimizer,
              device: torch.device = device, epochs:int = 3):
  test_loss = []
  train_loss = []
  timer_start  = timer()

  for epoch in tqdm(range(epochs)):
    print(f'\nEpoch: {epoch+1}')

    train = train_step(dl_train, model, loss_fn, optimizer)
    test = test_step(dl_test, model, loss_fn)

    train_loss.append(train[0])
    test_loss.append(test[0])

    print(f'Train Loss: {train[0]:.2f} Acc: {train[1]:.2f}% | '\
    f'Test Loss: {test[0]:.2f} Acc:{test[1]:.2f}%')

  timer_end = timer()

  print(f'\nModel trained for {epochs} epochs in {timer_end - timer_start} seconds.')

  return (train_loss, test_loss)


def eval_model(model: torch.nn.Module, acc_fn,
               data: torch.utils.data.DataLoader):
  acc = 0
  with torch.inference_mode():
    for X, y in data:
      logit = model(X)
      pred = torch.softmax(logit, dim=1).argmax(dim=1)

      acc += acc_fn(y, pred)

    acc /= len(data)

  return {
      'model_name': model.__class__.__name__,
      'accuracy': acc,
  }

