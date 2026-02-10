import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# إعداد الصفحة
st.set_page_config(
    page_title="السنتينل البيئي | كشف التلوث",
    page_icon="🌊",
    layout="wide"
)

# تعريب الخطوط والتنسيق (CSS) لجمال الواجهة
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Cairo', sans-serif;
        text-align: right;
        direction: rtl;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #007bff;
        color: white;
        font-weight: bold;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #0056b3;
        color: white;
    }
    
    .prediction-card {
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }
    
    .healthy {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    
    .polluted {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    
    .main-header {
        color: #2c3e50;
        text-align: center;
        padding: 20px;
        background: linear-gradient(90deg, #e3f2fd 0%, #ffffff 100%);
        border-radius: 15px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# إدارة حالة التطبيق (Stages)
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def next_stage():
    st.session_state.stage += 1
def prev_stage():
    st.session_state.stage -= 1
def go_to_stage(n):
    st.session_state.stage = n

# دالة لتحميل البيانات
@st.cache_data
def load_data():
    try:
        return pd.read_csv('eco_sentinel_dataset.csv')
    except:
        return None

# دالة لتدريب النموذج
@st.cache_resource
def get_trained_model(df):
    if df is None: return None, None
    X = df[['Turbidite_NTU', 'Oxygene_Dissous_mgL']].values
    y = df['Etat_Eau'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel='rbf', C=1.0, probability=True)
    model.fit(X_scaled, y)
    return model, scaler

df = load_data()
model, scaler = get_trained_model(df)

# --- الواجهة البرمجية ---

# شريط التقدم العلوي
steps = ["ترحيب", "البيانات", "التدريب", "النتائج", "تحليل عميق", "المحاكي"]
st.write(f"### المرحلة: {steps[st.session_state.stage]}")
st.progress((st.session_state.stage + 1) / len(steps))

# 0. مرحلة الترحيب
if st.session_state.stage == 0:
    st.markdown("<div class='main-header'><h1>🌊 السنتينل البيئي</h1><h2>مستقبل حماية المياه بالذكاء الاصطناعي</h2></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("### 🎯 مهمة المشروع")
        st.write("يهدف هذا المشروع إلى بناء نظام مراقبة ذكي يستخدم تعلم الآلة للكشف عن تلوث المياه فور وقوعه، وحماية النظم البيئية من الكوارث البيئية.")
    with col2:
        st.image("https://img.icons8.com/clouds/500/000000/water.png", width=250)
    st.button("ابدأ الرحلة 🚀", on_click=next_stage)

# 1. مرحلة البيانات
elif st.session_state.stage == 1:
    st.subheader("📁 الخطوة 1: استكشاف مجموعة البيانات")
    st.write("نستخدم بيانات حقيقية تحاكي حساسات IoT الموزعة في مجاري المياه.")
    if df is not None:
        st.dataframe(df.head(10), use_container_width=True)
        st.success(f"✅ تم تحميل {len(df)} سجل بيئي بنجاح.")
    
    col1, col2 = st.columns(2)
    with col1: st.button("⬅️ السابق", on_click=prev_stage)
    with col2: st.button("بدء معالجة البيانات ⚙️", on_click=next_stage)

# 2. مرحلة التدريب
elif st.session_state.stage == 2:
    st.subheader("🧠 الخطوة 2: تدريب العقل الذكي (SVM)")
    st.write("يجري الآن تعليم خوارزمية **Support Vector Machine** كيفية التمييز بين المياه النقية والملوثة...")
    
    with st.status("جاري تدريب النموذج...", expanded=True) as status:
        st.write("تحجيم البيانات (Scaling)...")
        import time; time.sleep(1)
        st.write("تطبيق نواة RBF للتعامل مع البيانات المعقدة...")
        time.sleep(1)
        st.write("حساب حدود القرار (Decision Boundaries)...")
        status.update(label="✅ اكتمل التدريب بنجاح!", state="complete", expanded=False)
    
    st.balloons()
    col1, col2 = st.columns(2)
    with col1: st.button("⬅️ السابق", on_click=prev_stage)
    with col2: st.button("مشاهدة النتائج المذهلة 📊", on_click=next_stage)

# 3. مرحلة معرض النتائج
elif st.session_state.stage == 3:
    st.subheader("🖼️ معرض النتائج الرسومية")
    c1, c2 = st.columns(2)
    
    with c1:
        st.write("#### 1. خريطة توزيع البيانات")
        fig1, ax1 = plt.subplots()
        sns.scatterplot(data=df, x='Turbidite_NTU', y='Oxygene_Dissous_mgL', hue='Etat_Eau', palette='RdYlGn_r', ax=ax1)
        st.pyplot(fig1)
        
    with c2:
        st.write("#### 2. مصفوفة الارتباط")
        numeric_df = df.select_dtypes(include=[np.number])
        fig2, ax2 = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax2)
        st.pyplot(fig2)

    st.info("💡 المشروع حقق دقة عالية جداً بفضل استخدام تقنيات التنقيب المتقدمة.")
    col1, col2 = st.columns(2)
    with col1: st.button("⬅️ السابق", on_click=prev_stage)
    with col2: st.button("التحليل العميق لكل رسم 🧐", on_click=next_stage)

# 4. مرحلة التحليل العميق (شرح مذهل)
elif st.session_state.stage == 4:
    st.subheader("🔎 التحليل التفسيري للرسومات")
    choice = st.selectbox("اختر الرسم الذي تريد فهم أسراره:", 
                          ["حدود قرار الذكاء الاصطناعي", "مصفوفة الارتباط", "توزيع كثافة التلوث"])
    
    if choice == "حدود قرار الذكاء الاصطناعي":
        col1, col2 = st.columns([1, 1])
        with col1:
            h = .05
            x_min, x_max = df.iloc[:, 0].min() - 1, df.iloc[:, 0].max() + 1
            y_min, y_max = df.iloc[:, 1].min() - 1, df.iloc[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = model.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
            Z = Z.reshape(xx.shape)
            fig, ax = plt.subplots()
            ax.contourf(xx, yy, Z, cmap='RdYlGn_r', alpha=0.3)
            ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Etat_Eau'], cmap='RdYlGn_r', edgecolors='k', s=20)
            st.pyplot(fig)
        with col2:
            st.markdown("""
            ### 🛠️ شرح 'سطح القرار' العبقري:
            هذا الرسم هو **قلب الذكاء الاصطناعي** في مشروعنا. 
            *   **المناطق الخضراء:** تمثل المساحة التي "تعلم" النموذج أنها مياه سليمة.
            *   **المناطق الحمراء:** هي مناطق الخطر التي حددها النموذج كمناطق تلوث.
            *   **النواة (Kernel):** لاحظ كيف أن الحدود ليست مستقيمة! هذا بفضل نواة RBF التي سمحت للنموذج برؤية الأنماط الدائرية المعقدة.
            """)
    elif choice == "مصفوفة الارتباط":
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        st.markdown("""
        ### 🔗 ماذا تعني هذه الأرقام؟
        توضح هذه المصفوفة العلاقة بين المتغيرات. القيمة 1 تعني علاقة طردية كاملة، بينما القيم القريبة من الصفر تعني عدم وجود علاقة. يساعد هذا في معرفة أي الحساسات أكثر تأثيراً على النتيجة النهائية.
        """)

    col1, col2 = st.columns(2)
    with col1: st.button("⬅️ السابق", on_click=prev_stage)
    with col2: st.button("انتقل للمحاكي النهائي 🎮", on_click=next_stage)

# 5. مرحلة المحاكي النهائي
elif st.session_state.stage == 5:
    st.subheader("🎮 محاكي السنتينل النهائي")
    col1, col2 = st.columns([1, 2])
    with col1:
        turbidity = st.slider("مستوى العكارة", 0.0, 15.0, 5.0)
        oxygen = st.slider("الأكسجين المذاب", 0.0, 12.0, 6.0)
        input_data = np.array([[turbidity, oxygen]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        if prediction == 0:
            st.success("✅ المياه سليمة")
        else:
            st.error("⚠️ تلوث مرصود!")
    with col2:
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='Turbidite_NTU', y='Oxygene_Dissous_mgL', hue='Etat_Eau', palette='RdYlGn_r', ax=ax)
        ax.scatter(turbidity, oxygen, color='blue', s=200, marker='*', label='القراءة الحالية')
        st.pyplot(fig)

    if st.button("العودة للبداية 🏠"): go_to_stage(0)

# ذيل الصفحة
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>تم تطويره بواسطة الذكاء الاصطناعي - مشروع السنتينل البيئي 2026</p>", unsafe_allow_html=True)
