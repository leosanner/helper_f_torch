import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.model_selection import confusion_matrix
import random
from train.py import make_predictions


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


def plot_results(data, true_labels, pred_labels, label_names):
    '''Plot image true label x pred label, if true==pred-> green title, else red'''
  fig = plt.figure(figsize=(10,8))
  r = int(np.sqrt(len(data)))
  c = r

  for i in range(0, len(data)):
    plt.subplot(r, c, i+1)
    plt.imshow(data[i].squeeze(), cmap='grey')
    t_l = label_names[true_labels[i]]
    t_p = label_names[pred_labels[i]]

    if t_l == t_p:
      plt.title(f'True: {t_l} | Predict: {t_p}', c='g')

    else:
      plt.title(f'True: {t_l} | Predict: {t_p}', c='r')

    plt.axis(False)
  plt.tight_layout()
  plt.show()


def plot_cm(y_true, y_pred, classes,
            size=(7,7), color='Blues'):
  
  '''y_true, y_pred -> tensors, classes -> list of classes names'''

  cm = confusion_matrix(y_true=torch.cat(y_true),
                      y_pred=torch.cat(y_pred))

  plt.figure(figsize=size, dpi=300)
  ax = sns.heatmap(cm, xticklabels=classes, yticklabels=classes,
              cmap=color, annot=True, fmt='d', cbar=False,)

  for _, spine in ax.spines.items():
      spine.set_visible(True)
      spine.set_color('black')
      spine.set_linewidth(1)

  plt.title('Matriz de Confusão')
  plt.xlabel('Valores Preditos Pelo Modelo')
  plt.ylabel('Valores Reais')
  plt.xticks(rotation=45)
  plt.show()


def plot_predictions(model, data, true_labels, pred_labels, labels, k=9):
  test_samples = []
  test_labels = []

  for sample, label in random.sample(list(data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

  pred_labels = make_predictions(model=model, data=data).argmax(dim=1)
  plot_results(test_samples, test_labels, pred_labels, labels)
