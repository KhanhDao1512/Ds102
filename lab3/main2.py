import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_score, recall_score, f1_score

from src.data_loader import load_data
from src.preprocess import preprocess
from src.svm import SVM


IMG_SIZE = 128


# =========================
# LOAD DATA
# =========================

X_train, y_train = load_data("data/chest_xray/train", IMG_SIZE)
X_test, y_test = load_data("data/chest_xray/test", IMG_SIZE)

X_train = preprocess(X_train)
X_test = preprocess(X_test)

# convert label:
# NORMAL -> -1
# PNEUMONIA -> 1
y_train = 2 * y_train - 1
y_test = 2 * y_test - 1


# =========================
# CUSTOM SVM
# =========================

custom_model = SVM(
    lr=0.0005,
    lambda_=1,
    epochs=20
)

custom_model.fit(X_train, y_train)

custom_pred = custom_model.predict(X_test)

print("\n===== CUSTOM SVM =====")

print("Precision:", precision_score(y_test, custom_pred))
print("Recall:", recall_score(y_test, custom_pred))
print("F1:", f1_score(y_test, custom_pred))


# =========================
# SKLEARN SVM
# =========================

sk_model = LinearSVC(
    C=1.0,
    max_iter=3000
)

sk_model.fit(X_train, y_train)

sk_pred = sk_model.predict(X_test)

print("\n===== SKLEARN SVM =====")

print("Precision:", precision_score(y_test, sk_pred))
print("Recall:", recall_score(y_test, sk_pred))
print("F1:", f1_score(y_test, sk_pred))