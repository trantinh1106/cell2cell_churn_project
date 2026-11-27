"""
utils/preprocessing.py
-----------------------
Tiền xử lý dữ liệu NÂNG CAO cho bộ dữ liệu Cell2Cell Churn (51.047 dòng):
  1. Đọc dữ liệu thô (Loading data)
  2. Làm sạch & Chuẩn hoá dữ liệu (Data Cleaning & Ordinal Conversion)
  3. Feature Engineering ngành Viễn thông (Telecom Domain Features)
     - Tỷ lệ rớt mạng (Dropped Call Rate) & nghẽn mạng (Blocked Call Rate)
     - Cước phí & Áp lực vượt gói (Overage Ratio, Revenue Per Minute)
     - Tỷ lệ già hóa thiết bị (Equipment Age Ratio)
     - Độ phân kỳ sử dụng (Usage-Revenue Divergence)
     - Tần suất CSKH & Tỷ lệ giữ chân thuê bao (Customer Care & Retention)
  4. Mã hoá khu vực (Frequency Encoding cho ServiceArea - tránh mất dữ liệu vùng)
  5. Mã hoá biến phân loại (One-Hot Encoding có kiểm soát)
  6. Phân chia Train/Test (80/20 Stratified) TRƯỚC TIÊN chống Data Leakage
  7. Xử lý giá trị khuyết (Median Imputation fit trên Train)
  8. Gây nhiễu dữ liệu (Gaussian Noise Injection - Tăng cường tính chống chịu / Regularization)
  9. Xử lý Outlier & Chuẩn hoá (RobustScaler fit trên Train)
  10. Xử lý mất cân bằng dữ liệu đa phương pháp:
      - SMOTE (Synthetic Minority Over-sampling)
      - ADASYN (Adaptive Synthetic Sampling)
      - Random Under-sampling (Giảm mẫu đa số)
      - Class Weighting (trong mô hình)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

# Kiểm tra an toàn thư viện imbalanced-learn
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.under_sampling import RandomUnderSampler
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

from utils.logger import print_header


def load_data(path: str) -> pd.DataFrame:
    """Đọc file CSV gốc."""
    print_header("1. LOADING DATASET", width=70)
    df = pd.read_csv(path)
    print(f"Dataset Loaded Successfully from: {path}")
    print(f"Tổng số dòng: {df.shape[0]:,} | Tổng số cột: {df.shape[1]}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu cơ bản:
      - Loại bỏ CustomerID (mã định danh ngẫu nhiên)
      - Mã hoá cột mục tiêu Churn: Yes -> 1, No -> 0
      - Chuẩn hoá các biến nhị phân (Yes/No, Known/Unknown) về dạng 1/0
      - Chuyển CreditRating (1-Highest...7-Lowest) thành số thứ tự 1-7
      - Chuyển HandsetPrice thành biến số thực (Numeric)
    """
    df = df.copy()

    # Loại bỏ CustomerID
    if "CustomerID" in df.columns:
        df.drop(columns=["CustomerID"], inplace=True)

    # Mã hoá cột mục tiêu Churn
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Mã hoá các biến nhị phân Yes/No
    binary_cols = [
        "ChildrenInHH", "HandsetRefurbished", "HandsetWebCapable",
        "TruckOwner", "RVOwner", "BuysViaMailOrder", "RespondsToMailOffers",
        "OptOutMailings", "NonUSTravel", "OwnsComputer", "HasCreditCard",
        "NewCellphoneUser", "NotNewCellphoneUser", "OwnsMotorcycle",
        "MadeCallToRetentionTeam"
    ]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    # Chuẩn hoá biến Homeownership (Known: 1, Unknown: 0)
    if "Homeownership" in df.columns:
        df["Homeownership"] = df["Homeownership"].map({"Known": 1, "Unknown": 0})

    # Chuyển CreditRating dạng '1-Highest' -> lấy ký tự đầu làm số nguyên (1..7)
    if "CreditRating" in df.columns:
        df["CreditRating"] = pd.to_numeric(
            df["CreditRating"].astype(str).str[0], errors="coerce"
        )

    # Chuyển HandsetPrice từ chuỗi sang số (vd: '10', '30', '150', 'Unknown' -> NaN)
    if "HandsetPrice" in df.columns:
        df["HandsetPrice"] = pd.to_numeric(df["HandsetPrice"], errors="coerce")

    return df


def engineer_telecom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng hành vi và chất lượng dịch vụ chuyên sâu ngành viễn thông.
    Giúp mô hình có chiều sâu nghiệp vụ, tăng mạnh khả năng phân loại rời mạng.
    """
    df = df.copy()
    eps = 1e-5

    # 1. Tính tổng lưu lượng cuộc gọi ước tính
    total_calls = (
        df.get("ReceivedCalls", 0) +
        df.get("OutboundCalls", 0) +
        df.get("BlockedCalls", 0) +
        df.get("DroppedCalls", 0) + eps
    )

    # 2. Nhóm Chất lượng dịch vụ & Khó chịu của khách hàng (Service Frustration)
    if "DroppedCalls" in df.columns:
        df["Dropped_Call_Rate"] = df["DroppedCalls"] / total_calls
    if "BlockedCalls" in df.columns:
        df["Blocked_Call_Rate"] = df["BlockedCalls"] / total_calls
    if "CustomerCareCalls" in df.columns and "MonthsInService" in df.columns:
        df["CustomerCare_Per_Month"] = df["CustomerCareCalls"] / (df["MonthsInService"] + 1)

    # 3. Nhóm Chi phí & Sốc cước (Billing Dynamics)
    if "OverageMinutes" in df.columns and "MonthlyMinutes" in df.columns:
        df["Overage_Ratio"] = df["OverageMinutes"] / (df["MonthlyMinutes"] + 1)
    if "MonthlyRevenue" in df.columns and "MonthlyMinutes" in df.columns:
        df["Revenue_Per_Minute"] = df["MonthlyRevenue"] / (df["MonthlyMinutes"] + 1)
    if "PercChangeMinutes" in df.columns and "PercChangeRevenues" in df.columns:
        # Độ phân kỳ: Dùng ít đi nhưng cước vẫn cao hoặc ngược lại -> nguy cơ bỏ mạng
        df["Usage_Revenue_Divergence"] = df["PercChangeMinutes"] - df["PercChangeRevenues"]

    # 4. Nhóm Thiết bị & Vòng đời khách hàng (Equipment & Tenure)
    if "CurrentEquipmentDays" in df.columns and "MonthsInService" in df.columns:
        # Tỷ lệ thiết bị cũ so với thời gian gắn bó (giá trị cao -> khách muốn đổi máy mới)
        df["Equipment_Age_Ratio"] = df["CurrentEquipmentDays"] / (df["MonthsInService"] * 30.5 + eps)
    if "ActiveSubs" in df.columns and "UniqueSubs" in df.columns:
        df["Active_Sub_Ratio"] = df["ActiveSubs"] / (df["UniqueSubs"] + eps)

    # 5. Nhóm Giữ chân thuê bao (Retention)
    if "RetentionOffersAccepted" in df.columns and "RetentionCalls" in df.columns:
        df["Retention_Success_Rate"] = df["RetentionOffersAccepted"] / (df["RetentionCalls"] + eps)

    return df


def encode_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Mã hoá nâng cao:
      - Frequency Encoding cho ServiceArea: Giữ lại thông tin địa lý mà không bùng nổ số chiều
      - One-Hot Encoding cho các biến phân loại còn lại (PrizmCode, Occupation, MaritalStatus)
    """
    df = df.copy()

    # 1. Frequency Encoding cho ServiceArea (thay vì vứt bỏ hoàn toàn)
    if "ServiceArea" in df.columns:
        service_freq = df["ServiceArea"].value_counts(normalize=True).to_dict()
        df["ServiceArea_Freq"] = df["ServiceArea"].map(service_freq).fillna(0)
        df.drop(columns=["ServiceArea"], inplace=True)

    # 2. One-hot encoding cho các biến phân loại danh mục
    cat_cols = [c for c in ["PrizmCode", "Occupation", "MaritalStatus"] if c in df.columns]
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown").astype(str)

    print(f"\nCác cột phân loại được One-Hot Encoding: {cat_cols}")
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df_encoded


def inject_noise(X: pd.DataFrame, noise_level: float = 0.05, random_state: int = 42) -> pd.DataFrame:
    """
    Gây nhiễu dữ liệu bằng nhiễu Gauss (Gaussian Noise Injection):
      X_noisy = X + N(0, (noise_level * std(X))^2)
    - Áp dụng trên tập Train nhằm tăng cường tính bền vững (Robustness) và tránh quá khớp (Regularization).
    - Chỉ áp dụng trên các đặc trưng liên tục, không làm biến dạng các biến nhị phân (0/1).
    """
    if noise_level <= 0:
        return X

    rng = np.random.default_rng(random_state)
    X_noisy = X.copy()

    # Chỉ gây nhiễu cho các cột số liên tục có > 2 giá trị độc nhất
    continuous_cols = [
        col for col in X_noisy.columns 
        if pd.api.types.is_numeric_dtype(X_noisy[col]) and X_noisy[col].nunique() > 2
    ]

    for col in continuous_cols:
        col_std = X_noisy[col].std()
        if col_std > 0:
            noise = rng.normal(0, noise_level * col_std, size=len(X_noisy))
            X_noisy[col] = X_noisy[col] + noise
            # Giữ các giá trị không âm nếu bản chất cột không có số âm
            if (X[col] >= 0).all():
                X_noisy[col] = np.maximum(X_noisy[col], 0.0)

    print(f"  [✓] Gây nhiễu thành công (Gaussian Noise Level: {noise_level*100:.1f}%) trên {len(continuous_cols)} đặc trưng liên tục.")
    return X_noisy


def balance_dataset(X: pd.DataFrame, y: pd.Series, method: str = "none", random_state: int = 42):
    """
    Xử lý mất cân bằng dữ liệu với nhiều chiến lược:
      - 'none'       : Giữ nguyên tỷ lệ tự nhiên (dùng class weighting ở tầng mô hình).
      - 'smote'      : Tạo mẫu nhân tạo cho lớp thiểu số (SMOTE).
      - 'adasyn'     : Tạo mẫu nhân tạo thích ứng tập trung vào vùng biên khó học (ADASYN).
      - 'undersample': Giảm bớt mẫu của lớp đa số (Random Under-sampling).
    """
    method = str(method).lower().strip()
    if method == "none" or not method:
        return X, y, f"Không cân bằng (Tỷ lệ gốc: {dict(y.value_counts())})"

    if not HAS_IMBLEARN:
        msg = "Chưa cài thư viện 'imbalanced-learn'. Đang giữ nguyên tỷ lệ gốc (Khuyên dùng: pip install imbalanced-learn)"
        print(f"  [!] {msg}")
        return X, y, msg

    if method == "smote":
        sampler = SMOTE(random_state=random_state)
        desc = "SMOTE (Over-sampling)"
    elif method == "adasyn":
        sampler = ADASYN(random_state=random_state)
        desc = "ADASYN (Adaptive Over-sampling)"
    elif method in ["undersample", "under"]:
        sampler = RandomUnderSampler(random_state=random_state)
        desc = "Random Under-sampling"
    else:
        sampler = SMOTE(random_state=random_state)
        desc = f"SMOTE mặc định (do tham số '{method}' không hợp lệ)"

    X_bal, y_bal = sampler.fit_resample(X, y)
    info = f"{desc} -> Phân bố mới: {dict(pd.Series(y_bal).value_counts())}"
    print(f"  [✓] {info}")
    return X_bal, y_bal, info


def preprocess_pipeline(
    path: str,
    use_smote: bool = None,
    balance_method: str = "none",
    noise_level: float = 0.0,
    target_col: str = "Churn"
):
    """
    Pipeline hoàn chỉnh điều phối toàn bộ bước tiền xử lý NÂNG CAO:
      - Tích hợp Kỹ nghệ đặc trưng ngành viễn thông (Telecom Feature Engineering)
      - Điền khuyết (Median Imputation) fit trên Train
      - Gây nhiễu dữ liệu (Gaussian Noise Injection) trên tập Train
      - Chuẩn hoá RobustScaler (chống Outliers) fit trên Train
      - Cân bằng lớp đa phương pháp: SMOTE, ADASYN, Random Under-sampling
    """
    # Hỗ trợ tương thích ngược với use_smote cũ
    if use_smote is True and balance_method == "none":
        balance_method = "smote"

    df_raw = load_data(path)
    df_cleaned = clean_data(df_raw)

    print_header("2. PREPROCESSING & ADVANCED FEATURE ENGINEERING", width=70)
    
    # 1. Tạo đặc trưng mới chuyên sâu
    df_feat = engineer_telecom_features(df_cleaned)
    print("  [✓] Đã tạo thành công 8 đặc trưng nghiệp vụ viễn thông:")
    print("      - Dropped_Call_Rate, Blocked_Call_Rate, CustomerCare_Per_Month")
    print("      - Overage_Ratio, Revenue_Per_Minute, Usage_Revenue_Divergence")
    print("      - Equipment_Age_Ratio, Active_Sub_Ratio, Retention_Success_Rate")

    # 2. Mã hoá biến
    df_encoded = encode_data(df_feat, target_col)

    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    feature_names = X.columns.tolist()

    print(f"\nKích thước ma trận đặc trưng X sau tiền xử lý: {X.shape}")
    print(f"Kích thước vector nhãn y                     : {y.shape}")

    # 3. Phân chia Train/Test TRƯỚC TIÊN (80% Train, 20% Test) chống rò rỉ dữ liệu (Data Leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nChia tập dữ liệu (Stratified 80/20):")
    print(f"  + Tập Train: {X_train.shape[0]:,} mẫu")
    print(f"  + Tập Test : {X_test.shape[0]:,} mẫu")

    # 4. Điền khuyết (Imputation) bằng Median fit trên Train
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_names)
    print("  [✓] Xử lý giá trị khuyết thiếu (Median Imputation) hoàn tất.")

    # 5. Gây nhiễu dữ liệu (Gaussian Noise Injection) trên tập Train (nếu noise_level > 0)
    if noise_level > 0:
        X_train_imp = inject_noise(X_train_imp, noise_level=noise_level, random_state=42)

    # 6. Chuẩn hoá RobustScaler fit trên Train (Kháng Outlier tốt hơn StandardScaler)
    scaler = RobustScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=feature_names)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_imp), columns=feature_names)
    print("  [✓] Chuẩn hoá đặc trưng (RobustScaler - Kháng Outlier) hoàn tất.")

    # 7. Xử lý mất cân bằng lớp (Chỉ áp dụng trên tập Train!)
    print(f"\nPhân bố nhãn tập Train ban đầu: {dict(y_train.value_counts())}")
    X_train_raw_bal, y_train_raw_bal, bal_info_raw = balance_dataset(
        X_train_imp, y_train, method=balance_method, random_state=42
    )
    X_train_scaled_bal, y_train_scaled_bal, _ = balance_dataset(
        X_train_scaled, y_train, method=balance_method, random_state=42
    )

    return {
        "X_train_raw": X_train_raw_bal,
        "y_train_raw": y_train_raw_bal,
        "X_test_raw": X_test_imp,
        "X_train_scaled": X_train_scaled_bal,
        "y_train_scaled": y_train_scaled_bal,
        "X_test_scaled": X_test_scaled,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
        "imputer": imputer,
        "df_raw": df_cleaned,
        "balance_info": bal_info_raw,
        "noise_level": noise_level,
    }
