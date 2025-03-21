import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.model_selection import confusion_matrix
import random
from train.py import make_predictions

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def plot_random_data_img(data, dimension, labels, increase=0):
    '''data must be a list of tuples with tensors and labels, dimension must be a int, labels must be a list of strings'''

    plt.figure(figsize=(8+increase,8+increase))

    for i in range(1, dimension**2 +1 ):
        rand_n = np.random.randint(0, len(data))
        tensor, label_n = data[rand_n]
        tensor = tensor.permute(1, 2, 0)

        plt.subplot(dimension, dimension, i)
        plt.imshow(tensor)

        plt.title(labels[label_n])
        plt.axis(False)

    plt.tight_layout()
    plt.show()


def plot_fit_result(m:list, epochs):
    '''Plot Loss x Epoch train result for model fitting -> 
    apply funcion using model_fit(), m must be a list with 2 tensors'''
    
    plt.plot(range(1,epochs+1), [x.cpu().detach().numpy() for x in m[0]], label='Train', color='purple')
    plt.plot(range(1,epochs+1), [x.cpu().detach().numpy() for x in m[1]], label='Test', color='Red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()


def plot_results(model:torch.nn.Module,
                 data:list,
                 labels: list[str],
                 dimension: int = 3):
  
  tensor_list = []
  y_true = []
  y_pred = []

  random_data = random.choices(data, k=dimension**2)
  with torch.inference_mode():
      for X, y in random_data:
        pred = torch.softmax(model(X.to(device).unsqueeze(dim=0)), dim=1).argmax(dim=1).item()
        tensor_list.append(X)
        y_true.append(y)
        y_pred.append(pred)

  fig = plt.figure(figsize=(8,8))

  for idx, tensor in enumerate(tensor_list):
    tensor = tensor.permute(1, 2, 0)
    plt.subplot(dimension, dimension, idx+1)
    plt.imshow(tensor.squeeze(), cmap='grey')
    t_l = labels[y_true[idx]]
    t_p = labels[y_pred[idx]]

    if t_l == t_p:
      plt.title(f'True: {t_l} | Predict: {t_p}', c='g', pad=10)

    else:
      plt.title(f'True: {t_l} | Predict: {t_p}', c='r', pad=10)

    plt.axis(False)
  
  plt.subplots_adjust(wspace=0.9, hspace=0.8)
  plt.tight_layout()
  plt.show()