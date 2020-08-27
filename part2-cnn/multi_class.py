import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_file = 'rps/'
test_file = 'rps-test-set/'

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generation = train_datagen.flow_from_directory(
    train_file,
    target_size=(150,150),
    class_mode='categorical'
)

validation_generation = validation_datagen.flow_from_directory(
    test_file,
    target_size=(150,150),
    class_mode='categorical'
)


model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
model.fit_generator(train_generation,
                    epochs=20,
                    validation_data=validation_generation,
                    verbose=1
                    )
