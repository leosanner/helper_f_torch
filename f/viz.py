import matplotlib.pyplot as plt
import numpy as np

def plot_random_data_img(data, dimension, labels, increase=0):
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
    apply funcion using model_fit()'''
    
    plt.plot(range(1,epochs+1), [x.cpu().detach().numpy() for x in m[0]], label='Train', color='purple')
    plt.plot(range(1,epochs+1), [x.cpu().detach().numpy() for x in m[1]], label='Test', color='Red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.show()


def plot_results(data, true_labels, pred_labels, label_names):
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
