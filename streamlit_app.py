import streamlit as st
from PIL import Image
import os

FEATURE_DIR = "features"
FILTER_DIR = "filters"
IMAGE_PATH = "test_images/3.png"
MODEL_PATH = "saved_model"


@st.cache  # 👈 This function will be cached
def get_filter_features(model_path, img_path):
    from tensorflow import keras
    from visualize import get_filters_and_features, visualize_filters_features
    from infer import load_image

    model = keras.models.load_model(model_path)
    img = load_image(img_path)
    filters, features = get_filters_and_features(model, img)
    visualize_filters_features(filters, feature=False)
    visualize_filters_features(features, "features")
    return

def show_images(image_paths, use_column_width="always"):
    """
    """
    for path in image_paths:
        image = Image.open(path)
        caption=path.split("/")[-1]
        st.image(image, caption=caption, use_column_width=use_column_width)


with st.sidebar:
    st.write("Select test image")
    test_image = st.text_input('Example Default: test_images/3.png')
    st.write("Select what you want to display")
    image = st.checkbox('show test image')
    features = st.checkbox('show features')
    filters = st.checkbox('show filters')

if test_image:
    IMAGE_PATH=test_image
    get_filter_features(MODEL_PATH, IMAGE_PATH)
else:
    IMAGE_PATH="test_images/3.png"
    get_filter_features(MODEL_PATH, IMAGE_PATH)


if image:
        original_image = [IMAGE_PATH]
        st.header('Original Image')
        show_images(original_image, use_column_width="auto")

if features:
        feature_images = [os.path.join(FEATURE_DIR, filename) for filename in os.listdir(FEATURE_DIR)]
        st.header('Displaying features')
        show_images(feature_images)

if filters:
        filter_images = [os.path.join(FILTER_DIR, filename) for filename in os.listdir(FILTER_DIR)]
        st.header('Displaying filters')
        show_images(filter_images)

