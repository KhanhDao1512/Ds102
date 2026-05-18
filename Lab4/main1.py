from utils import load_and_split_data, evaluate_model, plot_feature_importance
from models.decisiontree import DecisionTree

# 1. Load Data
X_train, X_test, Y_train, Y_test, feature_names = load_and_split_data()

# 2. Build & Train
my_tree = DecisionTree(max_depth=7, min_samples_leaf=10, min_information_gain=0.01)
my_tree.train(X_train, Y_train)

# 3. In cây
print("\nTREE STRUCTURE:")
my_tree.print_tree()

# 4. Evaluate & Plot
evaluate_model(my_tree, X_train, Y_train, X_test, Y_test, "Custom Decision Tree")
plot_feature_importance(my_tree.feature_importances, feature_names, "Decision Tree - Feature Importance")