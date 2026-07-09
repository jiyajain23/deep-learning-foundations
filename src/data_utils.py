import os
import numpy as np
from PIL import Image

def load_devanagari_data(data_dir, img_height=32, img_width=32):
    classes = sorted(os.listdir(data_dir))
    num_classes = len(classes)

    X_list = []
    Y_list = []

    print(f"Loading data from {data_dir}")

    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)

        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            try:
                img = Image.open(img_path).convert('L').resize((img_width, img_height))

                img_array = np.array(img).flatten()

                img_array = img_array / 255.0

                X_list.append(img_array)
                Y_list.append(class_idx)
            except Exception as e:
                print(f"Skipping {img_path}: {e}")

    X = np.array(X_list).T
    m = len(Y_list)
    Y = np.array(Y_list)
    Y_one_hot = np.zeros((num_classes, m))
    Y_one_hot[Y, np.arange(m)] = 1

    print(f"Loaded {m} images. X shape: {X.shape}, Y shape: {Y_one_hot.shape}")

    return X, Y_one_hot, classes

def shuffle_data(X, Y):
    np.random.seed(42)
    m = X.shape[1]
    permutation = np.random.permutation(m)
    return X[:, permutation], Y[:, permutation]
