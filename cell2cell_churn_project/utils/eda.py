"""
utils/eda.py
-------------
Khám phá và trực quan hoá dữ liệu (Exploratory Data Analysis - EDA)
cho bộ dữ liệu Cell2Cell Churn (51.047 dòng, 58 cột).
Tự động lưu các biểu đồ phục vụ báo cáo tiểu luận vào output/eda/.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils.logger import print_header

OUTPUT_DIR = os.path.join("output", "eda")


def _ensure_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig_name: str):
    path = os.path.join(OUTPUT_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def run_full_eda(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Chạy toàn bộ quy trình EDA và in báo cáo ra console:
      - 5 dòng đầu, shape, thống kê missing values, mô tả thống kê
      - Phân phối biến mục tiêu (số lượng & tỷ lệ %)
      - Lưu biểu đồ phân phối và tương quan
    """
    _ensure_dir()
    print_header("EXPLORATORY DATA ANALYSIS (EDA)", width=70)

    print("\n--- 5 DÒNG ĐẦU TIÊN CỦA BỘ DỮ LIỆU ---")
    print(df.head(5))

    print("\n--- KÍCH THƯỚC DỮ LIỆU (SHAPE) ---")
    print(f"Số dòng: {df.shape[0]:,}, Số cột: {df.shape[1]}")

    print("\n--- TỔNG HỢP GIÁ TRỊ THIẾU (MISSING VALUES > 0) ---")
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        for col, count in null_cols.items():
            pct = (count / len(df)) * 100
            print(f"  + {col:<25}: {count:>6} dòng khuyết ({pct:.2f}%)")
    else:
        print("  Không có giá trị khuyết thiếu.")

    print("\n--- THỐNG KÊ MÔ TẢ CÁC BIẾN SỐ TIÊU BIỂU ---")
    sample_num_cols = [c for c in ["MonthlyRevenue", "MonthlyMinutes", "TotalRecurringCharge", 
                                  "MonthsInService", "CurrentEquipmentDays", "CustomerCareCalls"] if c in df.columns]
    print(df[sample_num_cols].describe())

    print(f"\n--- PHÂN BỐ NHÃN MỤC TIÊU ({target_col}) ---")
    counts = df[target_col].value_counts(dropna=False)
    percents = df[target_col].value_counts(normalize=True) * 100
    for val in counts.index:
        print(f"  + Lớp {val}: {counts[val]:,d} mẫu ({percents[val]:.2f}%)")

    # 1. Biểu đồ phân phối nhãn Churn
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=target_col, data=df, hue=target_col, palette="Set2", legend=False)
    for p in ax.patches:
        h = p.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(f"{int(h):,} ({pct:.1f}%)",
                    (p.get_x() + p.get_width() / 2, h),
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.title(f"Target Distribution: {target_col} (Cell2Cell)", fontsize=12, fontweight="bold")
    plt.xlabel("Customer Churn (0: No, 1: Yes)")
    plt.ylabel("Number of Customers")
    _save("class_distribution.png")

    # 2. Biểu đồ phân phối Doanh thu hàng tháng (MonthlyRevenue)
    if "MonthlyRevenue" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df[df["MonthlyRevenue"] < 250], x="MonthlyRevenue", hue=target_col, 
                     kde=True, palette="Set2", bins=30)
        plt.title("Monthly Revenue Distribution by Churn Status", fontsize=12, fontweight="bold")
        _save("monthlyrevenue_distribution.png")

    # 3. Biểu đồ phân phối Số phút gọi hàng tháng (MonthlyMinutes)
    if "MonthlyMinutes" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df[df["MonthlyMinutes"] < 2500], x="MonthlyMinutes", hue=target_col, 
                     kde=True, palette="Set2", bins=30)
        plt.title("Monthly Minutes Distribution by Churn Status", fontsize=12, fontweight="bold")
        _save("monthlyminutes_distribution.png")

    # 4. Biểu đồ số ngày dùng thiết bị hiện tại (CurrentEquipmentDays)
    if "CurrentEquipmentDays" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.histplot(data=df, x="CurrentEquipmentDays", hue=target_col, kde=True, palette="Set2", bins=30)
        plt.title("Current Equipment Days by Churn Status", fontsize=12, fontweight="bold")
        _save("currentequipmentdays_distribution.png")

    # 5. Biểu đồ tỷ lệ rời mạng theo cuộc gọi CSKH (CustomerCareCalls)
    if "CustomerCareCalls" in df.columns:
        plt.figure(figsize=(7, 4))
        care_grouped = df.copy()
        care_grouped["CareGroup"] = pd.cut(care_grouped["CustomerCareCalls"].fillna(0),
                                           bins=[-1, 0, 2, 5, 100],
                                           labels=["0 calls", "1-2 calls", "3-5 calls", ">5 calls"])
        churn_rate = care_grouped.groupby("CareGroup", observed=False)[target_col].mean() * 100
        ax = churn_rate.plot(kind="bar", color="salmon")
        plt.title("Churn Rate by Customer Care Calls (%)", fontsize=12, fontweight="bold")
        plt.ylabel("Churn Rate (%)")
        plt.xlabel("Customer Care Calls Group")
        plt.xticks(rotation=0)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}%",
                        (p.get_x() + p.get_width() / 2, p.get_height() / 2),
                        ha="center", va="center", color="white", fontweight="bold")
        _save("customercare_vs_churn.png")

    # 6. Biểu đồ tương quan (Top 15 đặc trưng tương quan mạnh nhất với Churn)
    num_df = df.select_dtypes(include=["number"]).copy()
    if target_col in num_df.columns:
        corr = num_df.corr()[target_col].drop(target_col)
        top15 = corr.abs().sort_values(ascending=False).head(15).index
        corr_matrix = num_df[list(top15) + [target_col]].corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True)
        plt.title("Correlation Heatmap: Top 15 Numerical Features vs Churn", fontsize=12, fontweight="bold")
        _save("correlation_heatmap.png")
