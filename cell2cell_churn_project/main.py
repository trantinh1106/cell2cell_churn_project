"""
main.py
--------
Chương trình điều phối toàn bộ luồng huấn luyện và đánh giá dự án:
Dự đoán rời mạng khách hàng viễn thông (Cell2Cell Telecom Churn - 51.047 dòng).

Các giai đoạn thực thi:
  1. LOADING & PREPROCESSING (Tiền xử lý, chia tập Train/Test chống Data Leakage)
  2. EXPLORATORY DATA ANALYSIS (EDA và trực quan hoá dữ liệu)
  3. TRAINING & EVALUATION (Huấn luyện 6 mô hình với class weighting tối ưu)
  4. MODEL COMPARISON (So sánh tổng thể, lưu file CSV và vẽ biểu đồ)

Cách chạy:
  python main.py
"""

import os
import sys

# Đảm bảo đường dẫn import tương đối chuẩn xác
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import start_logging, stop_logging, print_header
from utils.preprocessing import preprocess_pipeline
from utils import eda
from utils import evaluation

from models import (
    logistic_regression,
    svm,
    knn,
    decision_tree,
    random_forest,
    xgboost_model,
)

DATA_PATH = os.path.join("data", "cell2celltrain.csv")


def main():
    log_file = start_logging(os.path.join("output", "training_log.txt"))

    try:
        print_header("CHƯƠNG TRÌNH HUẤN LUYỆN DỰ ÁN CELL2CELL TELECOM CHURN", width=70)

        # -------------------------------------------------------------
        # BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU CHUẨN KHOA HỌC
        # -------------------------------------------------------------
        data = preprocess_pipeline(DATA_PATH, use_smote=False)

        # -------------------------------------------------------------
        # BƯỚC 2: KHÁM PHÁ & TRỰC QUAN HOÁ DỮ LIỆU (EDA)
        # -------------------------------------------------------------
        eda.run_full_eda(data["df_raw"])

        # -------------------------------------------------------------
        # BƯỚC 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ CÁC MÔ HÌNH HỌC MÁY
        # -------------------------------------------------------------
        model_specs = [
            ("Logistic Regression", logistic_regression.train, "scaled"),
            ("Linear SVM", svm.train, "scaled"),
            ("KNN", knn.train, "scaled"),
            ("Decision Tree", decision_tree.train, "raw"),
            ("Random Forest", random_forest.train, "raw"),
            ("XGBoost", xgboost_model.train, "raw"),
        ]

        results = []
        trained_models = {}

        for name, train_fn, data_kind in model_specs:
            if data_kind == "scaled":
                X_train, X_test = data["X_train_scaled"], data["X_test_scaled"]
                y_train = data["y_train_scaled"]
            else:
                X_train, X_test = data["X_train_raw"], data["X_test_raw"]
                y_train = data["y_train_raw"]

            model, result = evaluation.train_and_evaluate(
                name, train_fn, X_train, y_train, X_test, data["y_test"]
            )
            trained_models[name] = model
            results.append(result)

        # -------------------------------------------------------------
        # BƯỚC 4: BÁO CÁO TỔNG HỢP SO SÁNH
        # -------------------------------------------------------------
        evaluation.print_model_comparison(results)

        # -------------------------------------------------------------
        # BƯỚC 5: TRỰC QUAN HOÁ ĐẶC TRƯNG QUAN TRỌNG & LƯU MÔ HÌNH
        # -------------------------------------------------------------
        if "XGBoost" in trained_models:
            from models.xgboost_model import get_feature_importance
            import matplotlib.pyplot as plt
            import joblib
            feat_imp = get_feature_importance(trained_models["XGBoost"], data["feature_names"]).head(20)
            plt.figure(figsize=(10, 6))
            feat_imp[::-1].plot(kind="barh", color="teal")
            plt.title("Top 20 Đặc trưng quan trọng nhất (XGBoost) - Cell2Cell Churn", fontsize=12, fontweight="bold")
            plt.xlabel("Mức độ quan trọng (Feature Importance)")
            plt.tight_layout()
            feat_path = os.path.join("output", "top_feature_importance.png")
            plt.savefig(feat_path, dpi=120)
            plt.close()
            print(f"  [✓] Biểu đồ đặc trưng quan trọng đã lưu tại: {feat_path}")

            # Lưu mô hình tốt nhất phục vụ Streamlit Web App
            joblib.dump(trained_models["XGBoost"], os.path.join("output", "xgboost_model.pkl"))
            joblib.dump(data["feature_names"], os.path.join("output", "feature_names.pkl"))
            joblib.dump(data["imputer"], os.path.join("output", "imputer.pkl"))
            joblib.dump(data["scaler"], os.path.join("output", "scaler.pkl"))
            print("  [✓] Đã lưu mô hình XGBoost và cấu hình vào thư mục output/ phục vụ Streamlit App.")

        # -------------------------------------------------------------
        # HOÀN TẤT
        # -------------------------------------------------------------
        print_header("TẤT CẢ CÁC TÁC VỤ ĐÃ HOÀN TẤT THÀNH CÔNG", width=70)
        print("  [✓] Log tiến trình đã lưu tại        : output/training_log.txt")
        print("  [✓] Bảng so sánh mô hình đã lưu tại : output/model_comparison.csv")
        print("  [✓] Biểu đồ so sánh đã lưu tại      : output/model_accuracy.png")
        print("  [✓] Biểu đồ đặc trưng quan trọng    : output/top_feature_importance.png")
        print("  [✓] Biểu đồ EDA đã lưu tại          : output/eda/")
        print("  [✓] Ma trận nhầm lẫn đã lưu tại    : output/confusion_matrix/")
        print("  [✓] Đường cong ROC đã lưu tại       : output/roc_curve/")

    finally:
        stop_logging(log_file)


if __name__ == "__main__":
    main()
