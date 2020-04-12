import numpy as np
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPool2D, Conv2D, Dropout, BatchNormalization
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping

(X_train, y_train), (X_test, y_test) = load_data()

plt.imshow(X_train[0])
plt.show()

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))

X_train = X_train.astype('float32')/255.0
X_test = X_test.astype('float32')/255.0



print(X_train.shape)
print(y_train.shape)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(28,28, 1)))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax', kernel_initializer='he_normal'))

model.summary()
plot_model(model, 'model.png', show_shapes=True)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit(X_train, y_train, epochs=100, batch_size=256, verbose=1, validation_split=0.3, callbacks=[es])

plt.title('Learning curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.show()

model.save('model.h5')

loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy : " + str(acc))

yhat = np.argmax(model.predict([[X_train[0]]]))
print('Predicted value : ' + str(yhat) + ' and True value : ' + str(y_train[0]))
