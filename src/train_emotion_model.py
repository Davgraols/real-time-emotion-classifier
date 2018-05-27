import keras
from keras.models import Sequential
from keras.layers.core import Dense, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


train_path = "./../dataset/train_data"
test_path = "./../dataset/test_data"
valid_path = "./../dataset/valid_data"

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), classes=['neutral', 'smile'], batch_size=40)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224), classes=['neutral', 'smile'], batch_size=40)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224), classes=['neutral', 'smile'], batch_size=40)

images, labels = next(train_batches)

vgg16_model = keras.applications.vgg16.VGG16()
vgg16_model.summary()

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))

model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(train_batches, steps_per_epoch=10,
                    validation_data=valid_batches, validation_steps=4, epochs=5, verbose=2)

test_images, test_labels = next(test_batches)

score = model.evaluate(test_images, test_labels)
print('Test loss: ', score[0])
print('Test accuracy', score[1])

# save model as HDF5
model.save('./../models/detect_emo_model.h5')
print("Saved model to disk")