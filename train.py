import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

def get_data(size=None):
    """
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data("mnist.npz")
    if size:
        return (x_train[:size], y_train[:size]), (x_test[:size], y_test[:size])
    return (x_train, y_train), (x_test, y_test)

def get_dataset(features, labels):
    """
    """
    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).shuffle(100).batch(32)
    return dataset

def get_model(input_shape, num_classes=10, optimizer="adam", loss="sparse_categorical_crossentropy"):
    """
    """
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), activation="relu", padding="same", input_shape=input_shape),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(16, (3,3), padding="same", activation="relu"),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def train_model(model, train_datset, epochs=2):
    history = model.fit(train_datset, epochs=epochs)
    return model

def save_test_images(test_images, num=10, dir_path="test_images"):
    """
    """
    os.makedirs(dir_path, exist_ok=True)
    for i in range(num):
        img_path = os.path.join(dir_path, f"{i}.png")
        img = test_images[i]
        if len(img.shape) == 2:
            img = img[..., np.newaxis]
        keras.utils.save_img(img_path, img)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = get_data(100)
    save_test_images(x_test)
    
    model = get_model((28,28,1))
    train_datset= get_dataset(x_train, y_train)
    test_datset= get_dataset(x_test, y_test)

    model = train_model(model, train_datset)
    model.save("saved_model")

