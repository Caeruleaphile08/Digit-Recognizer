import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape, y_train.shape, X_test.shape, y_test.shape

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))  # Add an additional dense layer
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early Stopping and Model Checkpoint callbacks
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1)
mc = ModelCheckpoint("./bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
callbacks = [es, mc]

# Preprocess the data
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

# Convert classes to one-hot vectors
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Model training
history = model.fit(X_train, y_train, epochs=50, validation_split=0.3, callbacks=callbacks)

model_S = keras.models.load_model("D://digit_recogniser//bestmodel.h5")
score = model_S.evaluate(X_test,y_test)
print(f"The model accuracy is {score[1]}")
