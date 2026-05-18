from utils import load_and_split_data, evaluate_model, plot_feature_importance
from models.randomforest import RandomForestClassifier

X_train, X_test, Y_train, Y_test, feature_names = load_and_split_data()

# Build & Train Random Forest (Ví dụ dùng 10 cây)
my_rf = RandomForestClassifier(n_base_learner=10, max_depth=7, min_samples_leaf=10)
my_rf.train(X_train, Y_train)

evaluate_model(my_rf, X_train, Y_train, X_test, Y_test, "Custom Random Forest")
plot_feature_importance(my_rf.feature_importances, feature_names, "Random Forest - Feature Importance")