"""
models/knn.py
--------------
Mô hình K-láng giềng gần nhất (K-Nearest Neighbors Classifier).
Phân loại mẫu dữ liệu dựa trên đa số phiếu từ k điểm dữ liệu gần nhất trong không gian đặc trưng.
Yêu cầu dữ liệu đầu vào đã chuẩn hoá (X_train_scaled, X_test_scaled).
"""

from sklearn.neighbors import KNeighborsClassifier


def get_model(n_neighbors: int = 9, weights: str = "distance") -> KNeighborsClassifier:
    """Khởi tạo KNN với trọng số theo khoảng cách và xử lý đa luồng."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        n_jobs=-1,
    )


def train(X_train, y_train, **kwargs) -> KNeighborsClassifier:
    model = get_model(**kwargs)
    model.fit(X_train, y_train)
    return model
