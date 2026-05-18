import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection, metrics

def load_and_split_data(filepath="dataset/winequality-red.csv"):
    df = pd.read_csv(filepath, sep=';')
    X = np.array(df.iloc[:, :-1])
    Y = np.array(df.iloc[:, -1])
    feature_names = df.columns[:-1]

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        X, Y, test_size=0.25, random_state=0, stratify=Y
    )
    return X_train, X_test, Y_train, Y_test, feature_names

def evaluate_model(model, X_train, Y_train, X_test, Y_test, model_name="Model"):
    print(f"\n{'='*40}")
    print(f"EVALUATING: {model_name}")
    print(f"{'='*40}")

    # Train
    train_preds = model.predict(X_train)

    print("\n--- TRAIN PERFORMANCE ---")
    print(metrics.classification_report(Y_train, train_preds, zero_division=0))

    # Test
    test_preds = model.predict(X_test)

    print("\n--- TEST PERFORMANCE ---")
    print(metrics.classification_report(Y_test, test_preds, zero_division=0))

def plot_feature_importance(importances_dict, feature_names, title="Feature Importance"):
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances_dict)), list(importances_dict.values()), tick_label=feature_names)
    plt.title(title)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()