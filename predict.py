from keras.optimizers import RMSprop
from sklearn.metrics import confusion_matrix

from data import load_test_data
from train import CNN
import numpy as np
from MNIST.functions import plot_confusion_matrix

def load_model_network():

    X_test, Y_test = load_test_data()

    model = CNN('digitos.hdf5')
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    Y_pred = model.predict(X_test)

    # Convert predictions classes to one hot vectors
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_test, axis=1)

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes=range(10))




if __name__ == '__main__':
    """"""
    load_model_network()