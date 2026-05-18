from utils import load_and_split_data, evaluate_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, Y_train, Y_test, feature_names = load_and_split_data()

# Sklearn Decision Tree
sk_tree = DecisionTreeClassifier(max_depth=7, min_samples_leaf=10, random_state=0)
sk_tree.fit(X_train, Y_train)
evaluate_model(sk_tree, X_train, Y_train, X_test, Y_test, "Sklearn Decision Tree")

# Sklearn Random Forest
sk_rf = RandomForestClassifier(n_estimators=10, max_depth=7, min_samples_leaf=10, random_state=0)
sk_rf.fit(X_train, Y_train)
evaluate_model(sk_rf, X_train, Y_train, X_test, Y_test, "Sklearn Random Forest")