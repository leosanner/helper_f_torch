import matplotlib.pyplot as plt
import numpy as np
import torch


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
