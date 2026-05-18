import pandas as pd
import numpy as np
from collections import Counter
from models.treenode import TreeNode

class DecisionTree:
    def __init__(self, max_depth=4, min_samples_leaf=1, 
                 min_information_gain=0.0, numb_of_features_splitting=None,
                 amount_of_say=None):
        # Khởi tạo các siêu tham số (hyperparameters) điều kiện dừng của cây
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting
        self.amount_of_say = amount_of_say

    def _entropy(self, class_probabilities):
        # Tính độ nhiễu (Entropy) bằng công thức: -sum(p * log2(p))
        return sum(
            -p * np.log2(p)
            for p in class_probabilities
            if p > 0 # Bỏ qua p=0 để tránh lỗi log(0)
        )
    
    def _class_probabilities(self, labels):
        # Tính tỉ lệ % (xác suất) xuất hiện của từng nhãn trong tập dữ liệu
        total_count = len(labels)
        return [label_count/total_count for label_count in Counter(labels).values()]

    def _data_entropy(self, labels):
        # Tính mức độ nhiễu tổng thể của tập nhãn hiện tại
        return self._entropy(self._class_probabilities(labels))
    
    def _partition_entropy(self, subsets):
        # Tính Entropy trung bình có trọng số sau khi đã chia thành các nhóm con (subsets)
        total_count = sum(len(subset) for subset in subsets)
        return sum(
            self._data_entropy(subset) * len(subset) / total_count
            for subset in subsets
        )
    
    def _split(self, data, feature_val, feature_idx):
        # Tách dữ liệu thành 2 nhánh: Nhỏ hơn ngưỡng (group1) và lớn hơn/bằng ngưỡng (group2)
        threshold = data[:,feature_idx] < feature_val # Tạo mảng boolean (True/False)
        group1 = data[threshold]  # Lấy các dòng thỏa mãn True
        group2 = data[~threshold] # Lấy các dòng thỏa mãn False (dấu ~ là Not)
        return group1, group2
    
    def _select_features(self, data):
        # Lấy danh sách các cột đặc trưng (trừ cột nhãn cuối cùng)
        feature_idx = list(range(data.shape[1] - 1))

        # Chọn ngẫu nhiên 1 số lượng feature (Dùng cho thuật toán Random Forest)
        if self.numb_of_features_splitting == "sqrt":
            size = max(1, int(np.sqrt(len(feature_idx))))
            feature_idx_to_use = np.random.choice(feature_idx, size=size, replace=False)
        elif self.numb_of_features_splitting == "log":
            size = max(1, int(np.log2(len(feature_idx))))
            feature_idx_to_use = np.random.choice(feature_idx, size=size, replace=False)
        else:
            feature_idx_to_use = feature_idx # Dùng toàn bộ features cho Decision Tree thường

        return feature_idx_to_use

    def _find_best_splits(self, data):
        # Tìm cách chia (split) tốt nhất: Giúp Entropy giảm nhiều nhất
        min_part_entropy = float("inf") # Khởi tạo giá trị lớn vô cùng
        best_split = None
        feature_idx_to_use = self._select_features(data)

        # Duyệt qua từng cột feature được chọn
        for idx in feature_idx_to_use:
            unique_values = np.unique(data[:, idx]) # Lấy các giá trị ko trùng lặp
            
            if len(unique_values) <= 1: continue # Cột có 1 giá trị -> Không thể chia

            # Tính các điểm chính giữa (midpoints) làm ngưỡng chia (thresholds)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            # Thử từng ngưỡng chia xem cái nào tối ưu nhất
            for threshold in thresholds:
                g1, g2 = self._split(data, threshold, idx)

                if len(g1) == 0 or len(g2) == 0: continue # Bỏ qua nếu có nhánh rỗng

                # Tính tổng nhiễu sau khi chia
                part_entropy = self._partition_entropy([g1[:, -1], g2[:, -1]])

                # Lưu lại thông tin nếu đây là cách chia tốt nhất (nhiễu thấp nhất)
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    best_split = (g1, g2, idx, threshold, part_entropy)

        return best_split
    
    def _find_label_probs(self, data):
        # Trả về mảng xác suất tương ứng với tất cả các nhãn (labels) đã học ở tập Train
        labels_as_integers = data[:, -1].astype(int)
        total_labels = len(labels_as_integers)
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == label)[0] # Tìm vị trí các dòng có nhãn này
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities
    
    def _create_tree(self, data, current_depth):
        # Hàm đệ quy để xây dựng cây

        # Điều kiện dừng 1: Đạt độ sâu tối đa -> Trả về Node Lá (Leaf)
        if current_depth >= self.max_depth:
            label_probabilities = self._find_label_probs(data)
            return TreeNode(data=data, feature_idx=None, feature_val=None, 
                            prediction_probs=label_probabilities, information_gain=0)

        best_split = self._find_best_splits(data)

        # Điều kiện dừng 2: Không tìm được cách chia nào -> Trả về Node Lá
        if best_split is None:
            label_probabilities = self._find_label_probs(data)
            return TreeNode(data=data, feature_idx=None, feature_val=None, 
                            prediction_probs=label_probabilities, information_gain=0)

        # Mở gói kết quả (unpack) từ cách chia tốt nhất
        split_1, split_2, split_feature_idx, split_feature_val, split_entropy = best_split

        label_probabilities = self._find_label_probs(data)
        node_entropy = self._entropy(label_probabilities)
        
        # Information Gain = Nhiễu ban đầu - Nhiễu sau khi chia (Càng lớn càng tốt)
        information_gain = node_entropy - split_entropy

        # Tạo Node nội bộ (Internal Node) để lưu điều kiện chia
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)

        # Điều kiện dừng 3: Các nhánh con có quá ít dữ liệu (< min_samples_leaf)
        if split_1.shape[0] < self.min_samples_leaf or split_2.shape[0] < self.min_samples_leaf:
            return node

        # Điều kiện dừng 4: Lượng thông tin thu được quá thấp (< min_information_gain)
        if information_gain < self.min_information_gain:
            return node

        current_depth += 1
        # Đệ quy: Tự gọi lại hàm này để xây tiếp nhánh trái và nhánh phải
        node.left = self._create_tree(split_1, current_depth)
        node.right = self._create_tree(split_2, current_depth)

        return node
    
    def _predict_one_sample(self, X: np.array) -> np.array:
        # Chạy 1 mẫu dữ liệu qua cây bằng cách đi dọc xuống theo các điều kiện chia
        node = self.tree
        while node.left is not None or node.right is not None:
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right
        return node.prediction_probs # Trả về xác suất tại node lá cuối cùng chạm tới

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """ Huấn luyện mô hình """
        self.labels_in_train = np.unique(Y_train) # Lưu lại tất cả các loại nhãn có thể có
        # Gép X và Y thành 1 ma trận duy nhất để dễ xử lý (Y nằm ở cột cuối)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        # Kích hoạt hàm đệ quy tạo cây từ gốc (depth = 0)
        self.tree = self._create_tree(data=train_data, current_depth=0)

        # Tính Feature Importance (để biết biến nào quan trọng nhất)
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self._calculate_feature_importance(self.tree)
        # Chuẩn hóa về thang % (tổng = 1)
        self.feature_importances = {k: v / sum(self.feature_importances.values()) for k, v in self.feature_importances.items()}

    def predict_proba(self, X: np.array) -> np.array:
        """ Trả về mảng xác suất dự đoán """
        # np.apply_along_axis: Áp dụng hàm _predict_one_sample cho từng dòng (axis=1) của X_set
        pred_probs = np.apply_along_axis(self._predict_one_sample, 1, X)
        return pred_probs

    def predict(self, X: np.array):
        # Lấy xác suất
        pred_probs = self.predict_proba(X)
        # Lấy vị trí (index) của nhãn có xác suất cao nhất trong mỗi dòng
        pred_indices = np.argmax(pred_probs, axis=1)
        # Ánh xạ index ngược lại thành tên/giá trị nhãn thực tế
        preds = self.labels_in_train[pred_indices]
        return preds    
        
    def _print_recursive(self, node: TreeNode, level=0) -> None:
        # Đệ quy để in cây. level dùng để thụt lề (indentation)
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def _calculate_feature_importance(self, node):
        # Đệ quy: Cộng dồn mức độ đóng góp (information_gain) của mỗi feature ở mọi node nhánh
        if node is None: return
        if node.feature_idx is not None:
            self.feature_importances[node.feature_idx] += node.feature_importance
        self._calculate_feature_importance(node.left)
        self._calculate_feature_importance(node.right)