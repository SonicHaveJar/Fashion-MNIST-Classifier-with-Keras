import tensorflow as tf
import numpy as np
from random import randint
import h5py
import os

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = tf.keras.utils.normalize(x_train)
x_test = tf.keras.utils.normalize(x_test)


tf.set_random_seed(4141)


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer='Adagrad',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30)



val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)


n = randint(0, len(x_test))

prediction = model. predict([x_test])
prediction_NUM = np.argmax(prediction[n])

if (prediction_NUM == 0):
    print('Camiseta/T-shirt')
elif (prediction_NUM == 1):
    print('Pantalon/Trouser')
elif (prediction_NUM == 2):
    print('Jersey/Pullover')
elif (prediction_NUM == 3):
    print('Vestido/Dress')
elif (prediction_NUM == 4):
    print('Abrigo/Coat')
elif (prediction_NUM == 5):
    print('Sandalia/Sandal')
elif (prediction_NUM == 6):
    print('Camisa/Shirt')
elif (prediction_NUM == 7):
    print('Zapatilla deportiva/Sneaker')
elif (prediction_NUM == 8):
    print('Bolso/Bag')
else:
    print('Bota de tobillo/Ankle boot')



if not os.path.exists('Model'):
    os.makedirs('Model')
    
model.save('Model/mnist_clothes.model')

