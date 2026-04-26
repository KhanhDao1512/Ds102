import cv2
import numpy as np
import os
from tqdm import tqdm

def load_data(data_dir, img_size):
    X = []
    y = []

    for label in ['NORMAL', 'PNEUMONIA']:
        class_dir = os.path.join(data_dir, label)

        for img_name in tqdm(os.listdir(class_dir), desc=f"Loading {label}"):
            img_path = os.path.join(class_dir, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (img_size, img_size))
            X.append(img.flatten())
            y.append(0 if label == 'NORMAL' else 1)

    return np.array(X), np.array(y)