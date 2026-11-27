"""
models/decision_tree.py
------------------------
Mô hình Cây quyết định (Decision Tree Classifier).
Áp dụng class_weight='balanced' để tối ưu độ nhạy (Recall) cho lớp Churn.
"""

from sklearn.tree import DecisionTreeClassifier


def get_model(max_depth: int = 7, min_samples_leaf: int = 25, 
              class_weight: str = "balanced", random_state: int = 42) -> DecisionTreeClassifier:
    """Khởi tạo Decision Tree với class_weight='balanced' nhằm xử lý mất cân bằng lớp."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
    )


def train(X_train, y_train, **kwargs) -> DecisionTreeClassifier:
    model = get_model(**kwargs)
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model: DecisionTreeClassifier, feature_names):
    import pandas as pd
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
