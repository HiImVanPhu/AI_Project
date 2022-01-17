from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow import keras
import matplotlib.pyplot as plt

train_dir = "D:\HUSTer Senior\AI and Application\Project\Data\Train"
val_dir = "D:\HUSTer Senior\AI and Application\Project\Data\Val"
test_dir = "D:\HUSTer Senior\AI and Application\Project\Data\Test"

size = 32

train_gen = ImageDataGenerator(rescale=1. / 255, height_shift_range=0.1, width_shift_range=0.1, zoom_range=0.1,
                               rotation_range=0.2, shear_range=0.2, fill_mode='nearest', horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1. / 255)

train_set = train_gen.flow_from_directory(train_dir, target_size=(size, size), class_mode='categorical',
                                          batch_size=32, color_mode="grayscale")
val_set = train_gen.flow_from_directory(val_dir, target_size=(size, size), class_mode='categorical',
                                        batch_size=32, color_mode="grayscale")
test_set = train_gen.flow_from_directory(test_dir, target_size=(size, size), class_mode='categorical',
                                         batch_size=32, color_mode="grayscale")


def build_ic2(shape, classes):
    model = Sequential()
    # layer 1
    model.add(Conv2D(32, (5, 5), activation='relu',
                     input_shape=shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    # layer 2
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    # layer 3
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.5))
    # layer 4
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    ######
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    ######
    model.add(Dense(classes, activation='softmax'))
    model.summary()
    return model


model = build_ic2(shape=(size, size, 1), classes=2)
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(decay=1e-6), metrics=['accuracy'])

history = model.fit(train_set, validation_data=val_set, steps_per_epoch=len(train_set), validation_steps=len(val_set),
                    epochs=100, callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

model.save("model_tf.h5")
eva = model.evaluate(test_set)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# build_ic2((32, 32, 3), 2)