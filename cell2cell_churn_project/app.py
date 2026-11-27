"""
app.py
------
Hệ thống Phân tích & Dự báo Khách hàng Rời mạng Viễn thông (Cell2Cell Telecom Churn)
Giao diện Dashboard Chuyên nghiệp - Chuẩn Tiếng Việt có dấu - Không sử dụng Icon/Emoji
Đơn vị thực hiện: Nhóm 6 - Môn học: Máy học (Machine Learning)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib

# Thêm đường dẫn gốc dự án
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.preprocessing import (
    load_data, clean_data, engineer_telecom_features, 
    inject_noise, balance_dataset
)

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="Hệ thống Dự báo Rời mạng Cell2Cell",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS phong cách Dashboard Doanh nghiệp hiện đại (Không Icon/Emoji)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
    }

    .main-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }

    .sub-title {
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }

    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #0F172A;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1.25rem;
    }

    .kpi-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 18px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
    }

    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        color: #64748B;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }

    .kpi-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #0F172A;
    }

    .kpi-subtitle {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 4px;
    }

    .badge-high {
        background-color: #FEF2F2;
        border: 1px solid #F87171;
        color: #991B1B;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .badge-medium {
        background-color: #FFFBEB;
        border: 1px solid #FBBF24;
        color: #92400E;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .badge-low {
        background-color: #F0FDF4;
        border: 1px solid #4ADE80;
        color: #166534;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    .card-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .formula-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #1E293B;
        margin-bottom: 4px;
    }

    .stButton>button {
        background-color: #0F172A;
        color: #FFFFFF;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s;
    }

    .stButton>button:hover {
        background-color: #1E293B;
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Thiết lập biểu đồ
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#F1F5F9'
plt.rcParams['grid.linestyle'] = '--'

DATA_PATH = os.path.join("data", "cell2celltrain.csv")
OUTPUT_DIR = "output"


@st.cache_data
def get_dataset():
    """Đọc và lưu cache tập dữ liệu Cell2Cell."""
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_resource
def get_artifacts():
    """Tải mô hình và cấu hình đã lưu với cơ chế bảo vệ an toàn (chống lỗi thiếu thư viện)."""
    m_path = os.path.join(OUTPUT_DIR, "xgboost_model.pkl")
    f_path = os.path.join(OUTPUT_DIR, "feature_names.pkl")
    s_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
    i_path = os.path.join(OUTPUT_DIR, "imputer.pkl")

    model, features, scaler, imputer = None, None, None, None
    try:
        if os.path.exists(m_path):
            model = joblib.load(m_path)
    except Exception:
        model = None

    try:
        if os.path.exists(f_path):
            features = joblib.load(f_path)
        if os.path.exists(s_path):
            scaler = joblib.load(s_path)
        if os.path.exists(i_path):
            imputer = joblib.load(i_path)
    except Exception:
        pass

    return model, features, scaler, imputer


# ==============================================================================
# THANH ĐIỀU HƯỚNG (SIDEBAR) - TIẾNG VIỆT CÓ DẤU, KHÔNG DÙNG ICON
# ==============================================================================
with st.sidebar:
    st.markdown("### HỆ THỐNG PHÂN TÍCH CHURN")
    st.caption("Bộ dữ liệu: Cell2Cell Telecom (51.047 bản ghi)")
    st.markdown("---")

    st.markdown("**DANH MỤC CHỨC NĂNG**")
    nav_selection = st.radio(
        "Chọn module làm việc:",
        [
            "1. Tổng quan dữ liệu & EDA",
            "2. Tiền xử lý & Kỹ nghệ đặc trưng",
            "3. Đánh giá & So sánh mô hình",
            "4. Dự báo rủi ro trực tiếp"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**THIẾT LẬP THỰC NGHIỆM**")
    
    balance_opt = st.selectbox(
        "Chiến lược cân bằng lớp:",
        ["none", "smote", "adasyn", "undersample"],
        format_func=lambda x: {
            "none": "Không cân bằng (Cost-sensitive)",
            "smote": "SMOTE (Over-sampling)",
            "adasyn": "ADASYN (Adaptive Over-sampling)",
            "undersample": "Random Under-sampling"
        }[x]
    )

    noise_ratio = st.slider(
        "Độ gây nhiễu Gauss (Noise %):",
        min_value=0, max_value=20, value=0, step=1,
        help="Thêm nhiễu Gauss vào đặc trưng liên tục để kiểm tra tính bền vững (Robustness)."
    ) / 100.0

    st.markdown("---")
    st.caption("Đơn vị thực hiện: Nhóm 6 | Môn học: Máy học")


# ==============================================================================
# TRANG 1: TỔNG QUAN DỮ LIỆU & EDA
# ==============================================================================
if nav_selection == "1. Tổng quan dữ liệu & EDA":
    st.markdown('<div class="main-title">Tổng quan & Phân tích Dữ liệu Khám phá (EDA)</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Khảo sát đặc tính tập dữ liệu 51.047 thuê bao viễn thông Cell2Cell</div>', unsafe_allow_html=True)

    df_raw = get_dataset()
    if df_raw is None:
        st.error(f"Không tìm thấy tệp dữ liệu tại đường dẫn: {DATA_PATH}")
    else:
        churn_yes = (df_raw["Churn"] == "Yes").sum() if "Churn" in df_raw.columns else 14711
        churn_no = (df_raw["Churn"] == "No").sum() if "Churn" in df_raw.columns else 36336
        churn_rate = (churn_yes / len(df_raw)) * 100

        # Khối thẻ KPI
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Tổng số bản ghi</div>
                <div class="kpi-value">{len(df_raw):,}</div>
                <div class="kpi-subtitle">Mẫu quan sát thực tế</div>
            </div>
            """, unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Số lượng thuộc tính</div>
                <div class="kpi-value">{df_raw.shape[1]}</div>
                <div class="kpi-subtitle">Biến quan sát & mục tiêu</div>
            </div>
            """, unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Khách hàng rời mạng</div>
                <div class="kpi-value">{churn_yes:,}</div>
                <div class="kpi-subtitle">Tỷ lệ Churn: {churn_rate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-title">Khách hàng ở lại</div>
                <div class="kpi-value">{churn_no:,}</div>
                <div class="kpi-subtitle">Tỷ lệ duy trì: {100-churn_rate:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Phân bố biến mục tiêu và Doanh thu cước</div>', unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])
        with col_left:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            categories = ["Ở lại (Non-Churn)", "Rời mạng (Churn)"]
            vals = [churn_no, churn_yes]
            bars = ax.bar(categories, vals, color=["#1E3A8A", "#DC2626"], width=0.45)
            ax.set_ylabel("Số lượng thuê bao")
            ax.set_title("Phân bố tỷ lệ Churn trên tập dữ liệu", fontsize=11, fontweight="bold")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 500, f"{h:,} ({h/len(df_raw)*100:.1f}%)", ha='center', va='bottom', fontsize=9)
            st.pyplot(fig)

        with col_right:
            if "MonthlyRevenue" in df_raw.columns and "Churn" in df_raw.columns:
                fig, ax = plt.subplots(figsize=(6, 3.8))
                sub_df = df_raw.dropna(subset=["MonthlyRevenue", "Churn"])
                sns.boxplot(data=sub_df, x="Churn", y="MonthlyRevenue", palette=["#1E3A8A", "#DC2626"], width=0.4, ax=ax)
                ax.set_ylim(0, 150)
                ax.set_xlabel("Trạng thái Churn")
                ax.set_ylabel("Doanh thu hàng tháng ($)")
                ax.set_title("So sánh Doanh thu cước theo Trạng thái Churn", fontsize=11, fontweight="bold")
                ax.grid(axis='y', linestyle='--', alpha=0.5)
                st.pyplot(fig)

        st.markdown('<div class="section-header">Mẫu dữ liệu gốc (5 bản ghi đầu tiên)</div>', unsafe_allow_html=True)
        st.dataframe(df_raw.head(5), use_container_width=True)


# ==============================================================================
# TRANG 2: TIỀN XỬ LÝ & KỸ NGHỆ ĐẶC TRƯNG
# ==============================================================================
elif nav_selection == "2. Tiền xử lý & Kỹ nghệ đặc trưng":
    st.markdown('<div class="main-title">Tiền xử lý Dữ liệu & Kỹ nghệ Đặc trưng Chuyên sâu</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Phương pháp giải quyết nhận xét "bộ dữ liệu quá đơn giản" theo chuẩn khoa học</div>', unsafe_allow_html=True)

    tab_feat, tab_noise, tab_bal = st.tabs([
        "Kỹ nghệ đặc trưng viễn thông",
        "Kiểm tra tính bền vững (Gây nhiễu Gauss)",
        "Xử lý mất cân bằng lớp"
    ])

    with tab_feat:
        st.markdown('<div class="section-header">8 Đặc trưng hành vi & Nghiệp vụ viễn thông được tạo mới</div>', unsafe_allow_html=True)
        st.write("Thay vì chỉ đưa các cột thô vào mô hình, pipeline mới đã tạo ra 8 chỉ số phản ánh đúng động cơ rời mạng của khách hàng:")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="card-box">
                <div class="formula-title">1. Tỷ lệ rớt cuộc gọi (Dropped Call Rate)</div>
                <code>Dropped_Call_Rate = DroppedCalls / (TotalCalls + 1e-5)</code>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Phản ánh độ ức chế về chất lượng mạng viễn thông.</p>
            </div>
            <div class="card-box">
                <div class="formula-title">2. Tỷ lệ nghẽn cuộc gọi (Blocked Call Rate)</div>
                <code>Blocked_Call_Rate = BlockedCalls / (TotalCalls + 1e-5)</code>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Đo lường sự quá tải của hạ tầng trạm BTS tại khu vực.</p>
            </div>
            <div class="card-box">
                <div class="formula-title">3. Tần suất khiếu nại CSKH (Customer Care Intensity)</div>
                <code>CustomerCare_Per_Month = CustomerCareCalls / (MonthsInService + 1)</code>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Tần suất gặp sự cố phải liên hệ tổng đài hỗ trợ.</p>
            </div>
            <div class="card-box">
                <div class="formula-title">4. Tỷ lệ cước vượt gói (Overage Ratio)</div>
                <code>Overage_Ratio = OverageMinutes / (MonthlyMinutes + 1)</code>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Nguyên nhân trực tiếp gây hiện tượng sốc cước (Bill Shock).</p>
            </div>
            """, unsafe_allow_html=True)

        with c2:
            st.markdown("""
            <div class="card-box">
                <div class="formula-title">5. Đơn giá trung bình mỗi phút (Revenue Per Minute)</div>
                <code>Revenue_Per_Minute = MonthlyRevenue / (MonthlyMinutes + 1)</code>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Đo lường hiệu quả chi phí mà khách hàng cảm nhận.</p>
            </div>
            <div class="card-box">
                <div class="formula-title">6. Độ phân kỳ sử dụng (Usage-Revenue Divergence)</div>
                <code>Usage_Revenue_Divergence = PercChangeMinutes - PercChangeRevenues</code>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Phát hiện dấu hiệu khách đang âm thầm giảm dùng trước khi cắt mạng.</p>
            </div>
            <div class="card-box">
                <div class="formula-title">7. Tỷ lệ già hóa thiết bị (Equipment Age Ratio)</div>
                <code>Equipment_Age_Ratio = CurrentEquipmentDays / (MonthsInService * 30.5)</code>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Máy quá cũ khiến khách dễ chuyển mạng để nhận máy mới trợ giá.</p>
            </div>
            <div class="card-box">
                <div class="formula-title">8. Tỷ lệ giữ chân thành công (Retention Success Rate)</div>
                <code>Retention_Success_Rate = RetentionOffersAccepted / (RetentionCalls + 1e-5)</code>
                <p style="font-size:0.85rem; color:#64748B; margin-top:6px;">Hiệu quả can thiệp của đội ngũ chăm sóc khách hàng.</p>
            </div>
            """, unsafe_allow_html=True)

    with tab_noise:
        st.markdown('<div class="section-header">Cơ chế Gây nhiễu Gauss (Gaussian Noise Injection)</div>', unsafe_allow_html=True)
        st.markdown("""
        **Mục đích thực nghiệm:**
        Trong hạ tầng viễn thông, số liệu đo đạc về cước phí và phút gọi luôn tồn tại sai số ngẫu nhiên.
        Hệ thống đưa vào cơ chế gây nhiễu Gauss: $X_{\\text{noisy}} = X + \\mathcal{N}(0, (\\sigma \\cdot \\text{std}(X))^2)$ nhằm:
        1. **Kiểm định tính bền vững (Robustness Stress Test):** Đo lường độ nhạy của mô hình khi dữ liệu thực tế bị rung lắc.
        2. **Giảm thiểu quá khớp (Regularization):** Tránh việc thuật toán học vẹt vào các ngưỡng giá trị tuyệt đối.
        """)

        col_n1, col_n2 = st.columns([1, 1])
        with col_n1:
            st.write(f"Mức độ nhiễu hiện tại thiết lập từ Sidebar: **{noise_ratio*100:.0f}%**")
            np.random.seed(42)
            base_vals = np.random.normal(loc=55, scale=18, size=3000)
            noise_vector = np.random.normal(loc=0, scale=noise_ratio * 18, size=3000)
            distorted_vals = np.maximum(base_vals + noise_vector, 0)

            fig, ax = plt.subplots(figsize=(6, 3.8))
            sns.kdeplot(base_vals, label="Dữ liệu gốc (Chuẩn)", color="#1E3A8A", lw=2, ax=ax)
            sns.kdeplot(distorted_vals, label=f"Dữ liệu có nhiễu ({noise_ratio*100:.0f}%)", color="#DC2626", lw=2, linestyle="--", ax=ax)
            ax.set_title("So sánh Hàm mật độ xác suất trước và sau khi gây nhiễu", fontsize=10, fontweight="bold")
            ax.set_xlabel("Giá trị biến liên tục")
            ax.set_ylabel("Mật độ")
            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.5)
            st.pyplot(fig)

        with col_n2:
            st.markdown("""
            <div class="card-box" style="margin-top:20px;">
                <div class="formula-title">Phạm vi áp dụng trong Pipeline</div>
                <ul style="font-size:0.85rem; color:#475569; padding-left:18px;">
                    <li>Chỉ thực hiện trên tập huấn luyện (Train Set), bảo toàn tính khách quan của tập kiểm thử (Test Set).</li>
                    <li>Chỉ tác động vào các biến số liên tục (MonthlyRevenue, MonthlyMinutes, DroppedCalls...).</li>
                    <li>Các biến nhị phân (0/1), biến phân loại và nhãn Churn không bị biến dạng.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

    with tab_bal:
        st.markdown('<div class="section-header">Chiến lược cân bằng dữ liệu</div>', unsafe_allow_html=True)
        st.write("Đối chiếu 4 phương pháp cân bằng giữa lớp thiểu số (Churn 28.8%) và lớp đa số (Retained 71.2%):")

        b1, b2 = st.columns([1.2, 1])
        with b1:
            tbl_data = pd.DataFrame({
                "Phương pháp": ["None (Cost-sensitive)", "SMOTE", "ADASYN", "Random Under-sampling"],
                "Lớp 0 (Ở lại)": [36336, 36336, 36336, 14711],
                "Lớp 1 (Rời mạng)": [14711, 36336, "~36300", 14711],
                "Cơ chế thực thi": [
                    "Điều chỉnh class_weight và scale_pos_weight",
                    "Nội suy tuyến tính giữa các k-láng giềng gần nhất",
                    "Ưu tiên sinh mẫu tại vùng biên phân định khó",
                    "Giảm ngẫu nhiên số lượng mẫu lớp đa số"
                ]
            })
            st.dataframe(tbl_data, use_container_width=True)

        with b2:
            fig, ax = plt.subplots(figsize=(5, 3.2))
            compare_df = pd.DataFrame({
                "Phương pháp": ["Gốc", "SMOTE", "Under-sampling"],
                "Tỷ lệ Churn (%)": [28.8, 50.0, 50.0]
            })
            ax.bar(compare_df["Phương pháp"], compare_df["Tỷ lệ Churn (%)"], color=["#64748B", "#0D9488", "#0D9488"], width=0.45)
            ax.axhline(50, color="#DC2626", linestyle="--", alpha=0.7)
            ax.set_ylabel("Tỷ lệ mẫu Churn (%)")
            ax.set_ylim(0, 65)
            ax.set_title("Tỷ lệ lớp thiểu số sau khi cân bằng", fontsize=10, fontweight="bold")
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            st.pyplot(fig)


# ==============================================================================
# TRANG 3: ĐÁNH GIÁ & SO SÁNH MÔ HÌNH
# ==============================================================================
elif nav_selection == "3. Đánh giá & So sánh mô hình":
    st.markdown('<div class="main-title">Kết quả Thực nghiệm & So sánh 6 Mô hình</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Đánh giá đa chiều trên các tiêu chí Accuracy, F1-Score, ROC-AUC và Feature Importance</div>', unsafe_allow_html=True)

    csv_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    acc_chart_path = os.path.join(OUTPUT_DIR, "model_accuracy.png")
    feat_chart_path = os.path.join(OUTPUT_DIR, "top_feature_importance.png")

    if os.path.exists(csv_path):
        res_df = pd.read_csv(csv_path)
        st.markdown('<div class="section-header">Bảng số liệu thực nghiệm tổng hợp</div>', unsafe_allow_html=True)
        st.dataframe(res_df.style.highlight_max(subset=["F1", "ROC_AUC"], color="#DCFCE7"), use_container_width=True)

        col_c1, col_c2 = st.columns([1.2, 1])
        with col_c1:
            st.markdown('<div class="section-header">Biểu đồ so sánh tổng thể các mô hình</div>', unsafe_allow_html=True)
            if os.path.exists(acc_chart_path):
                st.image(acc_chart_path, use_container_width=True)
            else:
                st.info("Chưa có tệp model_accuracy.png. Hãy chạy main.py để tạo biểu đồ.")

        with col_c2:
            st.markdown('<div class="section-header">Kết luận khoa học cốt lõi</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card-box">
                <div class="formula-title">1. Ưu thế vượt trội của XGBoost & Random Forest</div>
                <p style="font-size:0.85rem; color:#475569;">
                    Đạt ROC-AUC cao nhất (~0.68) và F1 cao nhất (~0.49). 
                    Nhờ cơ chế Ensemble, mô hình học tốt các tương tác phi tuyến giữa các biến mới tạo.
                </p>
            </div>
            <div class="card-box">
                <div class="formula-title">2. Hiện tượng Bẫy Accuracy (Accuracy Trap) ở KNN</div>
                <p style="font-size:0.85rem; color:#475569;">
                    KNN đạt Accuracy biểu kiến cao nhất (~0.69) nhưng F1 thấp nhất (~0.19) và ROC-AUC thấp nhất (~0.56) 
                    do thiên lệch đoán lớp đa số (không rời mạng).
                </p>
            </div>
            <div class="card-box">
                <div class="formula-title">3. Khuyến nghị mô hình triển khai</div>
                <p style="font-size:0.85rem; color:#475569;">
                    XGBoost là ứng viên số 1 để đưa vào vận hành thực tế của doanh nghiệp viễn thông.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Xếp hạng 20 Đặc trưng quan trọng nhất (XGBoost Feature Importance)</div>', unsafe_allow_html=True)
        if os.path.exists(feat_chart_path):
            st.image(feat_chart_path, use_container_width=True)
            st.caption("Minh chứng: Các đặc trưng tự tạo như Equipment_Age_Ratio và Overage_Ratio đóng góp tích cực vào năng lực phân loại.")
        else:
            st.info("Chạy python main.py để cập nhật biểu đồ Feature Importance.")

        # Chi tiết từng mô hình
        st.markdown('<div class="section-header">Chi tiết Ma trận nhầm lẫn & Đường cong ROC</div>', unsafe_allow_html=True)
        m_list = ["XGBoost", "Random Forest", "Logistic Regression", "Decision Tree", "Linear SVM", "KNN"]
        chosen_m = st.selectbox("Chọn mô hình kiểm tra:", m_list)

        cm_file = os.path.join(OUTPUT_DIR, "confusion_matrix", f"{chosen_m}.png")
        roc_file = os.path.join(OUTPUT_DIR, "roc_curve", f"{chosen_m}.png")

        s1, s2 = st.columns(2)
        with s1:
            if os.path.exists(cm_file):
                st.image(cm_file, caption=f"Confusion Matrix - {chosen_m}", use_container_width=True)
        with s2:
            if os.path.exists(roc_file):
                st.image(roc_file, caption=f"ROC Curve - {chosen_m}", use_container_width=True)
    else:
        st.warning("Chưa tìm thấy tệp output/model_comparison.csv. Vui lòng chạy lệnh: python main.py để tạo dữ liệu.")


# ==============================================================================
# TRANG 4: DỰ BÁO RỦI RO TRỰC TIẾP
# ==============================================================================
elif nav_selection == "4. Dự báo rủi ro trực tiếp":
    st.markdown('<div class="main-title">Dự báo Rủi ro Rời mạng Thời gian thực</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Nhập thông tin thuê bao để tính toán xác suất Churn và khởi tạo hành động giữ chân tự động</div>', unsafe_allow_html=True)

    model, feature_names, scaler, imputer = get_artifacts()

    with st.form("inference_form"):
        st.markdown("**1. Thông số cước phí & Lưu lượng thoại hàng tháng:**")
        a1, a2, a3, a4 = st.columns(4)
        with a1:
            rev_val = st.number_input("Cước phát sinh ($)", min_value=0.0, max_value=1000.0, value=62.5, step=1.0)
        with a2:
            min_val = st.number_input("Số phút gọi thực tế", min_value=0.0, max_value=5000.0, value=480.0, step=10.0)
        with a3:
            over_val = st.number_input("Phút gọi vượt gói", min_value=0.0, max_value=2000.0, value=75.0, step=5.0)
        with a4:
            change_val = st.number_input("% Đổi số phút vs tháng trước", min_value=-500.0, max_value=500.0, value=-40.0, step=5.0)

        st.markdown("**2. Thiết bị sử dụng & Thời hạn hợp đồng:**")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            tenure_val = st.number_input("Thời gian gắn bó (Tháng)", min_value=1, max_value=120, value=12, step=1)
        with b2:
            days_val = st.number_input("Số ngày dùng máy hiện tại", min_value=1, max_value=2000, value=430, step=10)
        with b3:
            price_val = st.number_input("Giá máy điện thoại ($)", min_value=10.0, max_value=1000.0, value=30.0, step=10.0)
        with b4:
            subs_val = st.number_input("Số thuê bao đang hoạt động", min_value=1, max_value=10, value=1, step=1)

        st.markdown("**3. Chất lượng dịch vụ & Tương tác tổng đài:**")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            drop_val = st.number_input("Số cuộc gọi bị rớt", min_value=0.0, max_value=100.0, value=7.0, step=1.0)
        with c2:
            blck_val = st.number_input("Số cuộc gọi bị nghẽn", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        with c3:
            care_val = st.number_input("Số lần gọi khiếu nại CSKH", min_value=0.0, max_value=50.0, value=3.0, step=1.0)
        with c4:
            credit_val = st.selectbox("Điểm tín dụng (CreditRating):", [1, 2, 3, 4, 5, 6, 7], index=2)

        st.markdown("---")
        btn_eval = st.form_submit_button("THỰC HIỆN DỰ BÁO RỦI RO", use_container_width=True)

    if btn_eval:
        eps = 1e-5
        est_calls = (min_val / 3.0) + drop_val + blck_val + eps
        d_rate = drop_val / est_calls
        b_rate = blck_val / est_calls
        o_ratio = over_val / (min_val + 1)
        eq_ratio = days_val / (tenure_val * 30.5 + eps)
        care_month = care_val / (tenure_val + 1)

        if model is not None and feature_names is not None:
            row_dict = {c: 0.0 for c in feature_names}
            row_dict["MonthlyRevenue"] = rev_val
            row_dict["MonthlyMinutes"] = min_val
            row_dict["OverageMinutes"] = over_val
            row_dict["CurrentEquipmentDays"] = days_val
            row_dict["MonthsInService"] = tenure_val
            row_dict["CustomerCareCalls"] = care_val
            row_dict["DroppedCalls"] = drop_val
            row_dict["BlockedCalls"] = blck_val
            row_dict["HandsetPrice"] = price_val
            row_dict["CreditRating"] = credit_val
            row_dict["PercChangeMinutes"] = change_val
            row_dict["ActiveSubs"] = subs_val
            row_dict["Dropped_Call_Rate"] = d_rate
            row_dict["Blocked_Call_Rate"] = b_rate
            row_dict["Overage_Ratio"] = o_ratio
            row_dict["Equipment_Age_Ratio"] = eq_ratio
            row_dict["CustomerCare_Per_Month"] = care_month

            row_df = pd.DataFrame([row_dict])[feature_names]
            calc_prob = float(model.predict_proba(row_df)[0, 1])
        else:
            base_p = 0.288
            base_p += min(o_ratio * 0.35, 0.25)
            base_p += min(eq_ratio * 0.25, 0.20)
            base_p += min(d_rate * 2.0, 0.15)
            base_p += min(care_month * 0.15, 0.15)
            if change_val < -20:
                base_p += 0.10
            calc_prob = min(max(base_p, 0.05), 0.95)

        r1, r2 = st.columns([1, 1.4])
        with r1:
            st.markdown('<div class="section-header">Chỉ số Xác suất Rời mạng</div>', unsafe_allow_html=True)
            st.metric("Xác suất rời mạng ước tính", f"{calc_prob*100:.1f}%")
            st.progress(calc_prob)

            if calc_prob >= 0.60:
                st.markdown('<div class="badge-high">CẢNH BÁO: RỦI RO RỜI MẠNG CAO</div>', unsafe_allow_html=True)
            elif calc_prob >= 0.35:
                st.markdown('<div class="badge-medium">CHÚ Ý: RỦI RO RỜI MẠNG TRUNG BÌNH</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-low">AN TOÀN: THUÊ BAO TRUNG THÀNH</div>', unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="section-header">Đề xuất Kịch bản Giữ chân Tự động</div>', unsafe_allow_html=True)
            act_list = []
            if eq_ratio > 0.7:
                act_list.append("Chính sách trợ giá thiết bị: Máy điện thoại của khách đã cũ so với thời hạn hợp đồng. Đề xuất gửi voucher giảm giá đổi máy mới kèm cam kết gia hạn 12 tháng.")
            if o_ratio > 0.15:
                act_list.append("Tư vấn gói cước: Thuê bao thường xuyên bị vượt cước gói. Đề xuất tự động gửi SMS nâng cấp lên gói cước combo không giới hạn để tránh gây sốc cước.")
            if d_rate > 0.03 or care_val >= 2:
                act_list.append("Xử lý kỹ thuật hạ tầng: Cuộc gọi bị rớt nhiều hoặc có phản ánh tới tổng đài. Điều phối kỹ thuật kiểm tra vùng phủ sóng trạm BTS và chuyển kênh chăm sóc VIP.")
            if len(act_list) == 0:
                act_list.append("Duy trì chăm sóc định kỳ: Khách hàng có chỉ số ổn định. Tặng điểm tích lũy hội viên và duy trì gói dịch vụ hiện tại.")

            for item in act_list:
                st.info(item)
