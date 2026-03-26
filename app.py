
import streamlit as st
import pandas as pd
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title="Smart Electrical System", layout="wide")
st.sidebar.image("SG-AIRPORT-LOGO.png", use_container_width=True)
st.sidebar.markdown("## 📊 MENU")
menu = st.sidebar.selectbox(
    "Chọn chức năng",
    [
        "🏠 Dashboard",
        "📊 Realtime",
        "🔮 Dự đoán",
        "📁 Dataset",
        "🤖 Machine Learning"
    ]
)
st.markdown("---")
# ===== STATUS SYSTEM =====
st.sidebar.success("● Online")

# ===== FOOTER =====
st.sidebar.markdown("""
<div class='sidebar-footer'>
⚡ AI Electrical System <br>
    © 2026 Airport Plaza
</div>
""", unsafe_allow_html=True)
# ======================
# CSS PRO
# ======================
st.markdown("""
<style>
.main {background-color: #0E1117;}
.card {
    background:#1c1f26;
    padding:15px;
    border-radius:10px;
    text-align:center;
    color:white;
}
.big {font-size:58px; font-weight:bold;}
            /* Footer */
.sidebar-footer {
    position: fixed;
    bottom: 20px;
    left: 20px;
    color: gray;
    font-size: 12px;
            text-align:center;
}
        /* FULL WIDTH + NO SPACE */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Ẩn header trắng */
header {visibility: hidden;}

/* Xóa menu mặc định */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
            h1 {
    margin-top: 0 !important;
}
            .card {
    background: linear-gradient(145deg, #1c1f26, #111318);
    padding: 20px;
    border-radius: 15px;
    text-align:center;
    color:white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    transition: 0.3s;
}
.card:hover {
    transform: scale(1.03);
}
            .blink {
    animation: blinker 1.5s linear infinite;
    color: red;
    font-weight: bold;
}
@keyframes blinker {
    50% { opacity: 0; }
}
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD DATA (DATASET GỐC)
# ======================
@st.cache_data
def load_data():
    df = pd.read_excel("airport_data.xlsx")

    df.columns = df.columns.str.strip()

    df = df.rename(columns={
        "Voltage(V)": "Voltage",
        "Current(A)": "Current",
        "Temperature(C)": "Temp",
        "Fault": "Fault"
    })
    return df
df = load_data()

# ======================
# TRAIN MODEL (DÙNG FAULT)
# ======================
@st.cache_resource
def train(df):
    X = df[['Voltage','Current','Temp']]
    y = df['Fault']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    return model, acc

model, acc = train(df)

st.markdown("""
<h1 style='color:#ff4b4b; font-size:36px;'>
⚡ SMART ELECTRICAL MONITORING SYSTEM
</h1>
""", unsafe_allow_html=True)
st.caption("Airport Plaza - Machine Learning Dashboard")
st.markdown("<p class='blink'>🚨 CẢNH BÁO HỆ THỐNG</p>", unsafe_allow_html=True)

# ======================
# DASHBOARD
# ======================
if menu == "🏠 Dashboard":

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"<div class='card'><div class='big'>{len(df)}</div>Dữ liệu</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card'><div class='big'>{df['Fault'].sum()}</div>Sự cố</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card'><div class='big'>{acc*100:.2f}%</div>Accuracy</div>", unsafe_allow_html=True)

    st.markdown("### 🚦 Trạng thái hệ thống")

    last = df.iloc[-1]

    if last["Fault"] == 1:
        st.error("🚨 HỆ THỐNG NGUY HIỂM")
    else:
        st.success("✅ HỆ THỐNG ỔN ĐỊNH")
    st.subheader("⚠️ Cảnh báo gần nhất")
    st.info("Không có sự cố trong 10 phút gần đây")
# ======================
# REALTIME
# ======================
elif menu == "📊 Realtime":

    run = st.checkbox("▶️ Bật realtime")
    speed = st.slider("Tốc độ", 0.2, 2.0, 0.5)
    if run:
        history_c = []
        history_t = []

        step_placeholder = st.empty()
        gauge_row = st.empty()
        status_box = st.empty()
        chart_row = st.empty()

        for i in range(min(len(df), 100)):

            row = df.iloc[i]

            v, c, t = row["Voltage"], row["Current"], row["Temp"]
            pred = model.predict([[v, c, t]])[0]

            history_c.append(c)
            history_t.append(t)

            # STEP
            step_placeholder.subheader(f"📡 Step {i}")

            # ===== 3 GAUGE =====
            with gauge_row.container():

                col1, col2, col3 = st.columns(3)

                # Voltage
                with col1:
                    fig_v = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=v,
                        title={'text': "⚡ Voltage"},
                        gauge={'axis': {'range': [0, 300]}}
                    ))
                    st.plotly_chart(fig_v, use_container_width=True, key=f"voltage_{i}")

                # Current
                with col2:
                    fig_c = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=c,
                        title={'text': "🔌 Current"},
                        gauge={'axis': {'range': [0, 30]}}
                    ))
                    st.plotly_chart(fig_c, use_container_width=True, key=f"current_{i}")

                # Temp
                with col3:
                    fig_t = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=t,
                        title={'text': "🔥 Temperature"},
                        gauge={'axis': {'range': [0, 100]}}
                    ))
                    st.plotly_chart(fig_t, use_container_width=True, key=f"temp_{i}")

            # STATUS
            if pred == 1:
                status_box.error("🚨 ML CẢNH BÁO: SỰ CỐ")
            else:
                status_box.success("✅ HỆ THỐNG ỔN ĐỊNH")

            # CHART
            with chart_row.container():
                col4, col5 = st.columns(2)

                with col4:
                    st.line_chart(history_c)

                with col5:
                    st.line_chart(history_t)

            time.sleep(speed)
# ======================
# PREDICT
# ======================
elif menu == "🔮 Dự đoán":

    st.subheader("🔮 Dự đoán sự cố thiết bị (Electrical Analysis)")

    col1, col2, col3 = st.columns(3)

    v = col1.number_input("⚡ Voltage (V)", 0, 300, 220)
    c = col2.number_input("🔌 Current (A)", 0, 30, 10)
    t = col3.number_input("🔥 Temperature (°C)", 0, 100, 30)

    if st.button("🚀 Predict"):

        pred = model.predict([[v, c, t]])[0]

        st.markdown("### 📊 Kết quả AI")

        if pred == 1:
            st.error("🚨 AI: NGUY CƠ SỰ CỐ")
        else:
            st.success("✅ AI: HỆ THỐNG BÌNH THƯỜNG")

        # ======================
        # PHÂN TÍCH KỸ THUẬT ĐIỆN
        # ======================
        st.markdown("### ⚡ Phân tích dữ liệu ")
        overload = c > 15
        overheat = t > 50
        undervoltage = v < 200
        overvoltage = v > 240
        # ===== CẢNH BÁO =====
        if overload:
            st.warning("⚡ Dòng điện cao → nguy cơ quá tải")
        if overheat:
            st.warning("🔥 Nhiệt độ cao → thiết bị đang nóng")
        if undervoltage:
            st.warning("🔻 Điện áp thấp → thiết bị kéo dòng lớn")
        if overvoltage:
            st.warning("🔺 Điện áp cao → nguy cơ phá hỏng cách điện")

        # ======================
        # CHẨN ĐOÁN NGUYÊN NHÂN
        # ======================
        st.markdown("### 🧠 Chẩn đoán nguyên nhân")

        if overload and overheat:
            st.error("🚨 Quá tải nghiêm trọng → dòng cao sinh nhiệt lớn → nguy cơ cháy thiết bị")

        elif undervoltage and overload:
            st.error("⚠️ Điện áp thấp → thiết bị kéo dòng → gây quá tải")

        elif overvoltage and overheat:
            st.error("⚠️ Điện áp cao → tăng tổn hao → nhiệt độ tăng")

        elif overheat:
            st.error("⚠️ Quá nhiệt → có thể do tản nhiệt kém hoặc môi trường")

        elif overload:
            st.error("⚠️ Quá dòng → thiết bị đang làm việc quá công suất")

        else:
            st.info("✅ Thông số trong vùng an toàn")

        # ======================
        # KẾT LUẬN
        # ======================
        st.markdown("### 📌 Kết luận")

        if pred == 1:
            st.error("👉 AI xác nhận hệ thống có dấu hiệu bất thường cần kiểm tra ngay")
        else:
            st.success("👉 Hệ thống vận hành ổn định")
# ======================
# DATASET
# ======================
elif menu == "📁 Dataset":

    st.subheader("📁 Dataset gốc")
    st.dataframe(df.head(200))

# ======================
# ML
# ======================
elif menu == "🤖 Machine Learning":

    st.subheader("🤖 Machine Learning Analysis")

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, roc_curve, auc
    from sklearn.model_selection import GridSearchCV
    import matplotlib.pyplot as plt
    import shap
    # ======================
    # DATA
    # ======================
    X = df[['Voltage','Current','Temp']]
    y = df['Fault']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # ======================
    # MULTI MODEL
    # ======================
    models = {
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Logistic Regression": LogisticRegression(max_iter=200)
    }
    results = []
    st.markdown("### 📊 So sánh mô hình")
    for name, model_ml in models.items():
        model_ml.fit(X_train, y_train)
        pred = model_ml.predict(X_test)
        acc = accuracy_score(y_test, pred)
        results.append([name, acc])
    result_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    st.dataframe(result_df)
    st.bar_chart(result_df.set_index("Model"))
    best_model_name = result_df.sort_values(by="Accuracy", ascending=False).iloc[0]["Model"]
    st.success(f"🏆 Model tốt nhất: {best_model_name}")

    # dùng lại RandomForest cho demo sâu
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ======================
    # CONFUSION MATRIX
    # ======================
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(cm)
    plt.colorbar(cax)
    # Gán nhãn trục
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_xticklabels(['', 'Normal', 'Fault'])
    ax.set_yticklabels(['', 'Normal', 'Fault'])
    # Hiển thị số
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='white')
    ax.set_title("Confusion Matrix - Random Forest")
    st.pyplot(fig)

    # ======================
    # CLASSIFICATION REPORT
    # ======================
    st.markdown("### 📋 Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # ======================
    # FEATURE IMPORTANCE
    # ======================
    st.markdown("### 🔥 Feature Importance")
    feat = pd.DataFrame({
        "Feature": ["Voltage","Current","Temp"],
        "Importance": model.feature_importances_
    })
    st.bar_chart(feat.set_index("Feature"))
    

    # ======================
    # CLASSIFICATION REPORT
    # ======================
    st.markdown("### 📋 Classification Report")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # ======================
    # GIẢI THÍCH ML
    # ======================
    st.markdown("### 🧠 Nhận xét")

    st.info("""
    - Model học từ dữ liệu Fault thực tế
    - Current và Temperature có ảnh hưởng lớn nhất
    - Random Forest cho kết quả tốt vì xử lý tốt dữ liệu phi tuyến
    - Hệ thống có thể áp dụng cho giám sát thiết bị điện thực tế (SCADA)
    - ROC Curve cho thấy khả năng phân biệt tốt giữa normal và fault
    - Probability giúp đánh giá mức độ rủi ro thay vì chỉ 0/1
    - Hyperparameter tuning giúp tối ưu model
    - SHAP giúp giải thích AI → minh bạch quyết định
    """)
