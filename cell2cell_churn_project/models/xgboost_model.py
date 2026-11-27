"""
models/xgboost_model.py
--------------------------
Mô hình XGBoost (Extreme Gradient Boosting Classifier).
Áp dụng scale_pos_weight=2.47 tương ứng tỷ lệ mất cân bằng lớp (71.2% / 28.8%),
giúp tăng cường độ nhạy (Recall) đối với nhóm khách hàng có nguy cơ rời mạng.
"""

from xgboost import XGBClassifier


def get_model(n_estimators: int = 150, max_depth: int = 5,
              learning_rate: float = 0.05, scale_pos_weight: float = 2.47,
              random_state: int = 42) -> XGBClassifier:
    """Khởi tạo XGBoost tối ưu tốc độ và cân bằng lớp."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric="logloss",
        n_jobs=-1,
    )


def train(X_train, y_train, **kwargs) -> XGBClassifier:
    model = get_model(**kwargs)
    model.fit(X_train, y_train)
    return model


def get_feature_importance(model: XGBClassifier, feature_names):
    import pandas as pd
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
