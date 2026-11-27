"""
models/svm.py
--------------
Mô hình Support Vector Machine (Linear SVM).
Sử dụng LinearSVC kết hợp class_weight='balanced' tối ưu tốc độ và cân bằng lớp.
Yêu cầu dữ liệu đầu vào đã chuẩn hoá (X_train_scaled, X_test_scaled).
"""

from sklearn.svm import LinearSVC


def get_model(C: float = 1.0, class_weight: str = "balanced", random_state: int = 42) -> LinearSVC:
    """Khởi tạo Linear SVM nhanh, hỗ trợ siêu phẳng phân tách cân bằng trọng số lớp."""
    return LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=3000,
        random_state=random_state,
        dual="auto",
    )


def train(X_train, y_train, **kwargs) -> LinearSVC:
    model = get_model(**kwargs)
    model.fit(X_train, y_train)
    return model
