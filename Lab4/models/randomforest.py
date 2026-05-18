import numpy as np
from models.decisiontree import DecisionTree # Đổi tên file import cho khớp với file Decision Tree của bạn

class RandomForestClassifier():
    # Mô hình Rừng ngẫu nhiên (Tập hợp nhiều Cây quyết định)
    def __init__(self, n_base_learner=10, max_depth=5, min_samples_leaf=1, min_information_gain=0.0, \
                 numb_of_features_splitting="sqrt", bootstrap_sample_size=None) -> None:
        # numb_of_features_splitting nên mặc định là "sqrt" cho Random Forest
        self.n_base_learner = n_base_learner                     # Số lượng cây
        self.max_depth = max_depth                               # Độ sâu tối đa của mỗi cây
        self.min_samples_leaf = min_samples_leaf                 # Số mẫu tối thiểu ở node lá
        self.min_information_gain = min_information_gain         # Ngưỡng chia cắt
        self.numb_of_features_splitting = numb_of_features_splitting # Số cột ngẫu nhiên khi chia
        self.bootstrap_sample_size = bootstrap_sample_size       # Kích thước tập dữ liệu lấy mẫu con

    def _create_bootstrap_samples(self, X, Y) -> tuple:
        # Lấy mẫu ngẫu nhiên có hoàn lại (Bootstrap) để tạo tập Train riêng cho từng cây
        bootstrap_samples_X = []
        bootstrap_samples_Y = []

        for i in range(self.n_base_learner):
            if not self.bootstrap_sample_size:
                self.bootstrap_sample_size = X.shape[0] # Mặc định lấy bằng số lượng tập gốc
            
            # Chọn ngẫu nhiên index (có cho phép lặp lại: replace=True)
            sampled_idx = np.random.choice(X.shape[0], size=self.bootstrap_sample_size, replace=True)
            bootstrap_samples_X.append(X[sampled_idx])
            bootstrap_samples_Y.append(Y[sampled_idx])

        return bootstrap_samples_X, bootstrap_samples_Y

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        # Huấn luyện toàn bộ Rừng
        self.labels_in_train = np.unique(Y_train) # [FIXED] Lưu lại nhãn gốc (VD: 3, 4, 5, 6...)
        bootstrap_samples_X, bootstrap_samples_Y = self._create_bootstrap_samples(X_train, Y_train)

        self.base_learner_list = []
        # Tạo và huấn luyện từng cây con (Base Learner)
        for base_learner_idx in range(self.n_base_learner):
            base_learner = DecisionTree(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, \
                                        min_information_gain=self.min_information_gain, 
                                        numb_of_features_splitting=self.numb_of_features_splitting)
            
            base_learner.train(bootstrap_samples_X[base_learner_idx], bootstrap_samples_Y[base_learner_idx])
            self.base_learner_list.append(base_learner)

        # Tính toán mức độ quan trọng của các đặc trưng (Feature Importances)
        self.feature_importances = self._calculate_rf_feature_importance(self.base_learner_list)

    def _predict_proba_w_base_learners(self,  X_set: np.array) -> list:
        # Gom kết quả dự đoán (xác suất) của toàn bộ các cây
        pred_prob_list = []
        for base_learner in self.base_learner_list:
            pred_prob_list.append(base_learner.predict_proba(X_set))
        return pred_prob_list

    def predict_proba(self, X: np.array) -> np.array:
        # Tính xác suất trung bình từ tất cả các cây
        base_learners_pred_probs = self._predict_proba_w_base_learners(X)
        
        # [FIXED & OPTIMIZED] Gộp mảng 3D và tính trung bình theo trục 0 (trục các cây) 
        # Nhanh hơn rất nhiều so với dùng vòng lặp for duyệt từng observation
        pred_probs = np.mean(base_learners_pred_probs, axis=0)
        return pred_probs

    def predict(self, X: np.array) -> np.array:
        # Lấy nhãn có xác suất trung bình cao nhất
        pred_probs = self.predict_proba(X)
        pred_indices = np.argmax(pred_probs, axis=1) # Trả về vị trí (index) của nhãn
        
        # [FIXED] Map index ngược lại về nhãn gốc để trả kết quả đúng
        preds = self.labels_in_train[pred_indices]
        return preds
    
    def _calculate_rf_feature_importance(self, base_learners):
        # Tính trung bình Feature Importance của tất cả các cây
        feature_importance_dict_list = []
        for base_learner in base_learners:
            feature_importance_dict_list.append(base_learner.feature_importances)

        # Lấy value và tính trung bình
        feature_importance_list = [list(x.values()) for x in feature_importance_dict_list]
        average_feature_importance = np.mean(feature_importance_list, axis=0)

        # [FIXED] Trả về kiểu Dictionary để tương thích với code main.py vẽ biểu đồ
        return {i: val for i, val in enumerate(average_feature_importance)}