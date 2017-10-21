

import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding


np.random.seed(2)


# Preparacao dos dados

# manipulacao de graficos com matploit
sns.set(style='white', context='notebook', palette='deep')

# X_train, Y_train, x_test, y_test

def normalizacao(data):
    return data / 255.0


def create_train_data():
    """
    :return: X_train, Y_train
    """
    print('running create_train_data() ')
    train = pd.read_csv(os.getcwd()+"/MNIST/data/input/train.csv")

    Y_train = train["label"]

    # Drop 'label' column
    X_train = train.drop(labels=["label"], axis=1) # (42000, 784)

    # liberando mais espaco
    del train

    # exibe a quantidade de classes que serao exibidas
    Y_train.value_counts()
    g = sns.countplot(Y_train)
    plt.show()

    #  verificando os valores nulos e ausentes
    X_train.isnull().any().describe()


    # Normaliza os dados
    X_train = normalizacao(X_train)


    # muda a forma em 3 dimensoes
    X_train = X_train.values.reshape(-1,28,28,1) # (42000, 28, 28, 1)


    # Codificaçao de rotulo
    Y_train = to_categorical(Y_train, num_classes=10)

    g = plt.imshow(X_train[1][:,:,0])
    plt.show()

    return X_train, Y_train


def create_teste_data():
    """
    :return:
    """
    test = pd.read_csv("../input/test.csv")


def load_train_data():
    """
    :return:
    """

def load_test_data():
    """
    :return:
    """

if __name__ == '__main__':
    create_train_data()