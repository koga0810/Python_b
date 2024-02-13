from msilib.schema import Binary
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np

train_data_path = "C:\\work\\Classifier\\dataset\\train"
test_data_path = "C:\\work\\Classifier\\dataset\\test"

train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(directory=train_data_path,
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='binary'
                                                    )

test_generator = test_datagen.flow_from_directory(directory=train_data_path,
                                                    target_size=(150,150),
                                                    batch_size=32,
                                                    class_mode='binary'
                                                    )
print(train_generator.class_indices)

model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5,5), input_shape=(150, 150,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = optimizers.SGD(learning_rate=0.1)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

history = model.fit(train_generator, epochs=40, verbose=1, validation_data=(test_generator), steps_per_epoch=4000/32, validation_steps=4000/32 )

model.save('model.h5')

loss = history.history['loss']
val_loss = history.history['val_loss']

learning_count = len(loss) + 1

plt.plot(range(1, learning_count),loss,marker='+',label='loss')
plt.plot(range(1, learning_count),val_loss,marker='.',label='val_loss')
plt.legend(loc = 'best', fontsize=10)
plt.xlabel('learning_count')
plt.ylabel('loss')
plt.show()

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.plot(range(1, learning_count),accuracy,marker='+',label='accuracy')
plt.plot(range(1, learning_count),val_accuracy,marker='.',label='val_accuracy')
plt.legend(loc = 'best', fontsize=10)
plt.xlabel('learning_count')
plt.ylabel('accuracy')
plt.show()


