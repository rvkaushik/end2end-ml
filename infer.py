import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_image(path, target_size=(28,28)):
    img = keras.utils.load_img(path, grayscale=True, target_size=target_size)
    img = keras.preprocessing.image.img_to_array(img)
    img = img[np.newaxis, ...]
    return img

def predict(model_path, img_path):
    """
    """
    img = load_image(img_path)
    model = keras.models.load_model(model_path)
    return np.argmax(model.predict(img))

if __name__ == "__main__":
    model_path = "saved_model"
    img_path = "test_images/3.png"
    result = predict(model_path, img_path)
    print(result)
