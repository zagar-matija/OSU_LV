import numpy as np
from tensorflow import keras
from keras import layers
from keras.datasets import cifar10
from keras.utils import to_categorical
from matplotlib import pyplot as plt
import tensorboard
import os


# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]), plt.yticks([])
    plt.imshow(X_train[i])

plt.show()


# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32') / 255.0
X_test_n = X_test.astype('float32') / 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train, dtype="uint8")
y_test = to_categorical(y_test, dtype="uint8")

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32, 32, 3)))
model.add(layers.Conv2D(filters=32, kernel_size=(
    3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(
    3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(
    3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
NAME = "cnn_dropout"
tboard_log_dir = os.path.join("logs", NAME)

my_callbacks = [
    keras.callbacks.TensorBoard(log_dir=tboard_log_dir,
                                update_freq=100),
    keras.callbacks.EarlyStopping(monitor="val_loss",
                                  patience=5, verbose=1)

]

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train_n,
          y_train,
          epochs=10,
          batch_size=64,
          callbacks=my_callbacks,
          validation_split=0.1)


score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')


# 1. zad
# preciznost na skupu za učenje: 0.8998
# Tocnost na testnom skupu podataka: 72.34
# tijekom učenja je došlo do overfitanja

# 2. zad
# Tocnost na testnom skupu podataka: 73.85 (ali s 10 epoha a ne 20 kao u prvom)
# accuracy: 0.8948
