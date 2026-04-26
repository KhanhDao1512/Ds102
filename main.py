import os
import numpy as np
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocess import preprocess
from src.svm import SVM
from src.metrics import classification_metrics

IMG_SIZE = 128
CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def load_or_cache(data_dir, prefix):
    X_path = os.path.join(CACHE_DIR, f"{prefix}_X.npy")
    y_path = os.path.join(CACHE_DIR, f"{prefix}_y.npy")

    if os.path.exists(X_path) and os.path.exists(y_path):
        print(f"Loading cached {prefix}...")
        X = np.load(X_path)
        y = np.load(y_path)
    else:
        print(f"Processing {prefix}...")
        X, y = load_data(data_dir, IMG_SIZE)
        X = preprocess(X)

        # 🔥 FIX QUAN TRỌNG: convert label
        y = 2*y - 1

        np.save(X_path, X)
        np.save(y_path, y)

    return X, y


# ❗ FORCE CLEAR CACHE nếu nghi ngờ sai
def clear_cache():
    import shutil
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
        os.makedirs(CACHE_DIR)


# 👉 Uncomment dòng này nếu muốn reset cache
# clear_cache()


# Load data
X_train, y_train = load_or_cache("data/chest_xray/train", "train")
X_test, y_test = load_or_cache("data/chest_xray/test", "test")


# Model
model = SVM(
    lr=0.0001,
    lambda_=0.1,
    epochs=20
)

# Train
model.fit(X_train, y_train)


# 🔍 DEBUG cực quan trọng
y_pred = model.predict(X_test)

print("\nPrediction distribution:")
print(np.unique(y_pred, return_counts=True))

print("\nLabel distribution:")
print(np.unique(y_test, return_counts=True))


# Evaluate
results = classification_metrics(y_test, y_pred)

print("\nEvaluation:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")


# 📉 Plot loss
plt.plot(model.loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()