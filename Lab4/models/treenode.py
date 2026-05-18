import numpy as np

class TreeNode():
    # Nút trong Cây quyết định
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data                         # Dữ liệu tại nút
        self.feature_idx = feature_idx           # Cột (feature) để chia
        self.feature_val = feature_val           # Ngưỡng (threshold) chia
        self.prediction_probs = prediction_probs # Xác suất dự đoán
        self.information_gain = information_gain # Mức tăng thông tin
        
        self.feature_importance = self.data.shape[0] * self.information_gain # Độ quan trọng (số mẫu * IG)
        
        self.left = None  # Nhánh trái (< ngưỡng)
        self.right = None # Nhánh phải (>= ngưỡng)

    def node_def(self) -> str:
        # Tạo chuỗi mô tả để in cây
        if (self.left or self.right):
            # Nút rẽ nhánh (Internal Node)
            return f"NODE | Information Gain = {self.information_gain} | Split IF X[{self.feature_idx}] < {self.feature_val} THEN left O/W right"
        else:
            # Nút lá (Leaf Node) - Thống kê nhãn
            unique_values, value_counts = np.unique(self.data[:,-1], return_counts=True)
            output = ", ".join([f"{value}->{count}" for value, count in zip(unique_values, value_counts)])            
            
            return f"LEAF | Label Counts = {output} | Pred Probs = {self.prediction_probs}"