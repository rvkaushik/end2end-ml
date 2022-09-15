from math import ceil
import os
from tensorflow import keras
import matplotlib.pyplot as plt


from infer import load_image

def get_filters_and_features(model, img):
    """
    """
    ids = [i for i, layer in enumerate(model.layers) if "conv" in layer.name]
    print(f"conv layers ids --- {ids}")
    
    filters = []
    for i in ids:
        filter_ = model.layers[i].get_weights()[0]
        filter_min, filter_max = filter_.min(), filter_.max()
        filter_ = (filter_ - filter_min)/(filter_max - filter_min)
        filters.append(filter_)

    outputs = [model.layers[i].output for i in ids] # outputs for tf model api
    model = keras.models.Model(inputs=model.inputs, outputs=outputs)
    features = model.predict(img)

    return (filters, features)

def visualize_filters_features(filters, path="filters", feature=True):
    """
    """
    os.makedirs(path, exist_ok=True)
    cols = 6
    for i in range(len(filters)):
        num_filters = filters[i].shape[-1]
        rows = ceil(num_filters/cols)
        st=1
        for _ in range(rows):
            for _ in range(cols):
                if st==num_filters:
                    break
                ax = plt.subplot(rows, cols, st)
                ax.set_xticks([])
                ax.set_yticks([])
                if not feature:
                    plt.imshow(filters[i][:,:,0,st-1], cmap='gray')
                else:
                    plt.imshow(filters[i][0, :,:,st-1], cmap='gray')
                st+=1
        plt.savefig(f"{path}/conv_layer_{i+1}.png", bbox_inches="tight")


if __name__ == "__main__":
    model_path = "saved_model"
    img_path = "test_images/3.png"
    img = load_image(img_path)
    model = keras.models.load_model(model_path)
    filters, features = get_filters_and_features(model, img)
    print(features[0].shape)
    visualize_filters_features(filters, feature=False)
    visualize_filters_features(features, "features")

