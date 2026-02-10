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

# دالة لتحميل البيانات (مخزنة في الذاكرة لتوفير الجهد)
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('eco_sentinel_dataset.csv')
        return df
    except:
        return None

# دالة لتدريب النموذج (تتم مرة واحدة فقط ولا تستهلك المعالج لاحقاً)
@st.cache_resource
def get_trained_model(df):
    if df is None:
        return None, None
        
    X = df[['Turbidite_NTU', 'Oxygene_Dissous_mgL']].values
    y = df['Etat_Eau'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # استخدام C=1.0 و RBF لضمان الكفاءة
    model = SVC(kernel='rbf', C=1.0, probability=True)
    model.fit(X_scaled, y)
    
    return model, scaler

df = load_data()
model, scaler = get_trained_model(df)

# واجهة المستخدم
st.markdown("<div class='main-header'><h1>🌊 مشروع السنتينل البيئي</h1><h3>النظام الذكي للكشف عن تلوث المياه</h3></div>", unsafe_allow_html=True)

if model is None:
    st.error("❌ لم يتم العثور على ملف البيانات 'eco_sentinel_dataset.csv'. يرجى التأكد من وجود الملف في مجلد المشروع.")
else:
    # تقسيم الصفحة إلى أعمدة
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🛠️ إدخال بيانات الحساسات")
        st.write("قم بتعديل القيم أدناه لمحاكاة قراءات الحساسات:")
        
        turbidity = st.slider("مستوى العكارة (Turbidity - NTU)", 
                               min_value=0.0, max_value=15.0, value=5.0, step=0.1)
        
        oxygen = st.slider("الأكسجين المذاب (Dissolved Oxygen - mg/L)", 
                            min_value=0.0, max_value=12.0, value=6.0, step=0.1)
        
        st.markdown("---")
        predict_btn = st.button("تحليل جودة المياه 🔍")
        
        if predict_btn:
            # التحضير للتنبؤ
            input_data = np.array([[turbidity, oxygen]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0]
            
            if prediction == 0:
                st.markdown(f"""
                    <div class='prediction-card healthy'>
                        <h2>✅ مياه سليمة (Sain)</h2>
                        <p>احتمالية السلامة: {probability[0]:.2%}</p>
                        <p>جودة المياه ضمن المعايير الطبيعية.</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class='prediction-card polluted'>
                        <h2>⚠️ مياه ملوثة (Pollué)</h2>
                        <p>احتمالية التلوث: {probability[1]:.2%}</p>
                        <p>تنبيه: تم اكتشاف مؤشرات تلوث غير طبيعية!</p>
                    </div>
                    """, unsafe_allow_html=True)

    with col2:
        st.subheader("📊 رؤى البيانات والذكاء الاصطناعي")
        
        tab1, tab2 = st.tabs(["📈 توزيع البيانات", "⚙️ تفاصيل النموذج"])
        
        with tab1:
            st.write("توزيع قراءات العكارة مقابل الأكسجين في قاعدة البيانات:")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data=df, x='Turbidite_NTU', y='Oxygene_Dissous_mgL', hue='Etat_Eau', 
                            palette='RdYlGn_r', ax=ax)
            # إضافة النقطة الحالية
            ax.scatter(turbidity, oxygen, color='blue', s=200, marker='*', label='القراءة الحالية')
            ax.set_xlabel('العكارة (Turbidity)')
            ax.set_ylabel('الأكسجين المذاب (Oxygen)')
            ax.legend(['سليمة', 'ملوثة', 'القراءة الحالية'])
            st.pyplot(fig)
            
        with tab2:
            st.write("معلومات حول دقة النظام:")
            # حساب الدقة بشكل سريع للعرض
            X_test_scaled = scaler.transform(df[['Turbidite_NTU', 'Oxygene_Dissous_mgL']].values)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(df['Etat_Eau'], y_pred)
            
            st.metric("دقة النموذج (Accuracy)", f"{acc:.2%}")
            st.info("""
            يعتمد هذا النظام على نموذج **SVM (Support Vector Machine)** مع نواة **RBF**.
            تم اختيار هذا النموذج لقدرته العالية على التمييز بين البيانات غير الخطية المعقدة.
            """)
            
            # عرض مصفوفة الارتباط
            corr = df.corr()
            fig_corr, ax_corr = plt.subplots()
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax_corr)
            st.write("مصفوفة الارتباط بين الخصائص:")
            st.pyplot(fig_corr)

# ذيل الصفحة
st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>تم تطويره بواسطة الذكاء الاصطناعي - مشروع السنتينل البيئي 2026</p>", unsafe_allow_html=True)
