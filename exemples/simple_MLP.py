



from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout

import pandas as pd
import numpy as np
import cv2


# Read data
train = pd.read_csv('MNIST/data/train.csv')
labels = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('MNIST/data/test.csv').values).astype('float32')



#cv2.imshow("Numero", X_test[0])
#cv2.waitKey(0)
#cv2.destroyAllWindows()



# convert list of labels to binary class matrix
y_train = np_utils.to_categorical(labels)


# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

# normalization
mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = y_train.shape[1]

# Here's a Deep Dumb MLP (DDMLP)
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))


# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Training...")
model.fit(X_train, y_train, batch_size=16, epochs=10, validation_split=0.1,  verbose=1) #show_accuracy=True,

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)


write_preds(preds, "keras-mlp.csv")


"""
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
cv2.imwrite('numero0.png', X_test[0])
cv2.imwrite('numero1.png', X_test[1])
cv2.imwrite('numero2.png', X_test[2])
"""