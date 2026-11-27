"""
models/logistic_regression.py
-------------------------------
Mô hình Hồi quy Logistic (Logistic Regression Classifier).
Áp dụng class_weight='balanced' xử lý mất cân bằng lớp.
Yêu cầu dữ liệu đầu vào đã chuẩn hoá (X_train_scaled, X_test_scaled).
"""

from sklearn.linear_model import LogisticRegression


def get_model(C: float = 1.0, max_iter: int = 1000, 
              class_weight: str = "balanced", random_state: int = 42) -> LogisticRegression:
    """Khởi tạo Logistic Regression với class_weight='balanced'."""
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        random_state=random_state,
        solver="lbfgs",
    )


def train(X_train, y_train, **kwargs) -> LogisticRegression:
    model = get_model(**kwargs)
    model.fit(X_train, y_train)
    return model
