"""
models/random_forest.py
-------------------------
Mô hình Rừng ngẫu nhiên (Random Forest Classifier).
Phương pháp học tập hợp Bagging kết hợp nhiều cây quyết định, giảm phương sai.
Áp dụng class_weight='balanced' giúp tối ưu việc nhận diện khách hàng rời mạng.
"""

from sklearn.ensemble import RandomForestClassifier


def get_model(n_estimators: int = 150, max_depth: int = 12, 
              min_samples_leaf: int = 10, class_weight: str = "balanced",
              random_state: int = 42) -> RandomForestClassifier:
    """Khởi tạo Random Forest tối ưu cho bài toán mất cân bằng lớp."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )


def train(X_train, y_train, **kwargs) -> RandomForestClassifier:
    model = get_model(**kwargs)
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model: RandomForestClassifier, feature_names):
    import pandas as pd
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
