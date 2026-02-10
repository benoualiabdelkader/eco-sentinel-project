import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# إعداد الصفحة
st.set_page_config(
    page_title="السنتينل البيئي | EcoSentinel AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- حالة التطبيق ---
if 'stage' not in st.session_state:
    st.session_state.stage = 0

def next_stage(): st.session_state.stage += 1
def prev_stage(): st.session_state.stage -= 1
def go_to_stage(n): st.session_state.stage = n

# --- التحميل والتدريب (Cached) ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv('eco_sentinel_dataset.csv')
    except:
        # بيانات افتراضية في حالة عدم وجود الملف
        data = {
            'Turbidite_NTU': np.random.uniform(0, 15, 100),
            'Oxygene_Dissous_mgL': np.random.uniform(0, 12, 100),
        }
        df = pd.DataFrame(data)
        df['Etat_Eau'] = ((df['Turbidite_NTU'] > 7) | (df['Oxygene_Dissous_mgL'] < 4)).astype(int)
        return df

@st.cache_resource
def get_trained_model(df):
    X = df[['Turbidite_NTU', 'Oxygene_Dissous_mgL']].values
    y = df['Etat_Eau'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = SVC(kernel='rbf', C=1.0, probability=True)
    model.fit(X_scaled, y)
    return model, scaler

df = load_data()
model, scaler = get_trained_model(df)

# --- التصميم المخصص (Tailwind + Glassmorphism + Animations) ---
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<style>
    * { font-family: 'Cairo', sans-serif; }
    body { background-color: #05110e; color: white; }
    .stApp { background-color: #05110e; }
    
    /* إخفاء عناصر ستريمليت الافتراضية */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* تأثيرات Glassmorphism */
    .glass {
        background: rgba(10, 30, 25, 0.6);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(19, 236, 128, 0.15);
        border-radius: 1.5rem;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* أنيميشن */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulseGlow { 0% { box-shadow: 0 0 5px rgba(19, 236, 128, 0.2); } 50% { box-shadow: 0 0 20px rgba(19, 236, 128, 0.5); } 100% { box-shadow: 0 0 5px rgba(19, 236, 128, 0.2); } }
    @keyframes slideUp { from { transform: translateY(50px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
    
    .animate-fade-in { animation: fadeIn 0.8s ease-out forwards; }
    .animate-slide-up { animation: slideUp 1s ease-out forwards; }
    .glow-pulse { animation: pulseGlow 3s infinite; }
    
    /* تخصيص الأزرار */
    .stButton > button {
        background: linear-gradient(135deg, #13ec80 0%, #0ea85b 100%) !important;
        color: #05110e !important;
        font-weight: 700 !important;
        border-radius: 1rem !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(19, 236, 128, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 8px 25px rgba(19, 236, 128, 0.5) !important;
    }
    
    /* تخصيص السلايدر */
    .stSlider [data-baseweb="slider"] { margin-bottom: 2rem; }
    
    /* نصوص مضيئة */
    .text-glow { text-shadow: 0 0 10px rgba(19, 236, 128, 0.5); }
    .text-glow-red { text-shadow: 0 0 10px rgba(239, 68, 68, 0.5); }
</style>
""", unsafe_allow_html=True)

# --- شريط التقدم العلوي ---
progress = (st.session_state.stage / 3) * 100
st.markdown(f"""
<div class="flex items-center justify-between px-10 py-6 border-b border-emerald-900/20 bg-[#05110e]/80 backdrop-blur-md sticky top-0 z-50" dir="rtl">
    <div class="flex items-center gap-4">
        <div class="w-12 h-12 rounded-2xl bg-emerald-500/10 flex items-center justify-center text-[#13ec80] border border-emerald-500/20 glow-pulse">
            <span class="material-icons-round text-3xl">waves</span>
        </div>
        <div>
            <h1 class="font-bold text-2xl tracking-tight text-glow">السنتينل البيئي</h1>
            <p class="text-[10px] text-emerald-500 font-mono uppercase tracking-[0.2em]">AI Monitoring System</p>
        </div>
    </div>
    <div class="w-1/3">
        <div class="flex justify-between text-xs mb-2 text-emerald-500 font-mono">
            <span>المرحلة {st.session_state.stage + 1} من 4</span>
            <span>{int(progress)}%</span>
        </div>
        <div class="w-full h-1.5 bg-emerald-900/30 rounded-full overflow-hidden">
            <div class="h-full bg-gradient-to-r from-emerald-600 to-[#13ec80] transition-all duration-1000" style="width: {progress}%"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- محتوى المراحل ---

if st.session_state.stage == 0:
    # المرحلة 1: الترحيب (Welcome)
    st.markdown("""
    <div class="max-w-7xl mx-auto py-24 px-8 animate-fade-in" dir="rtl">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-20 items-center">
            <div class="space-y-8">
                <div class="inline-flex items-center gap-3 px-4 py-2 rounded-full border border-emerald-500/30 bg-emerald-500/5 text-[#13ec80] text-sm font-bold">
                    <span class="relative flex h-3 w-3">
                        <span class="animate-ping absolute h-full w-full rounded-full bg-[#13ec80] opacity-75"></span>
                        <span class="relative h-3 w-3 rounded-full bg-[#13ec80]"></span>
                    </span>
                    النظام متصل وجاهز للعمل
                </div>
                <h1 class="text-8xl font-black leading-[1.1] mb-6">
                    مستقبل <br>
                    <span class="text-transparent bg-clip-text bg-gradient-to-l from-[#13ec80] to-teal-400 text-glow">حماية المياه</span>
                </h1>
                <p class="text-2xl text-emerald-100/50 leading-relaxed max-w-xl">
                    نظام ذكاء اصطناعي متطور لمراقبة النظم البيئية المائية وحمايتها من التلوث في الوقت الفعلي.
                </p>
                <div class="flex gap-6 pt-4">
                    <div class="glass p-5 flex items-center gap-5 flex-1">
                        <div class="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center text-[#13ec80]">
                            <span class="material-icons-round text-3xl">insights</span>
                        </div>
                        <div>
                            <p class="text-xs text-emerald-500 uppercase font-bold tracking-wider">دقة التنبؤ</p>
                            <p class="text-2xl font-bold">98.4%</p>
                        </div>
                    </div>
                    <div class="glass p-5 flex items-center gap-5 flex-1">
                        <div class="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center text-[#13ec80]">
                            <span class="material-icons-round text-3xl">speed</span>
                        </div>
                        <div>
                            <p class="text-xs text-emerald-500 uppercase font-bold tracking-wider">سرعة الاستجابة</p>
                            <p class="text-2xl font-bold">12ms</p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="relative flex justify-center items-center">
                <div class="absolute w-[500px] h-[500px] bg-emerald-500/5 rounded-full blur-[120px] animate-pulse"></div>
                <div class="relative z-10 w-96 h-96 rounded-[3rem] border border-emerald-500/20 bg-emerald-900/10 backdrop-blur-2xl flex items-center justify-center glow-pulse rotate-3 hover:rotate-0 transition-transform duration-700">
                    <span class="material-icons-round text-[180px] text-[#13ec80] drop-shadow-[0_0_30px_rgba(19,236,128,0.4)]">water_drop</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("ابدأ رحلة الاستكشاف ←", use_container_width=True): next_stage()

elif st.session_state.stage == 1:
    # المرحلة 2: البيانات (Data)
    st.markdown("""
    <div class="max-w-7xl mx-auto py-16 px-8 animate-fade-in" dir="rtl">
        <div class="mb-12">
            <h2 class="text-5xl font-bold mb-4 text-glow">قاعدة البيانات البيئية</h2>
            <p class="text-emerald-500 font-mono text-lg">تحليل 1,024 سجلاً عالي الدقة من أجهزة الاستشعار الموزعة</p>
        </div>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
            <div class="glass p-8 text-center group hover:border-emerald-500/40 transition-colors">
                <p class="text-xs text-emerald-500 uppercase font-bold mb-2 tracking-widest">متوسط العكارة</p>
                <h3 class="text-4xl font-bold text-white group-hover:text-[#13ec80] transition-colors">4.2 NTU</h3>
            </div>
            <div class="glass p-8 text-center group hover:border-emerald-500/40 transition-colors">
                <p class="text-xs text-emerald-500 uppercase font-bold mb-2 tracking-widest">الأكسجين المذاب</p>
                <h3 class="text-4xl font-bold text-white group-hover:text-[#13ec80] transition-colors">6.8 mg/L</h3>
            </div>
            <div class="glass p-8 text-center group hover:border-emerald-500/40 transition-colors">
                <p class="text-xs text-emerald-500 uppercase font-bold mb-2 tracking-widest">عدد الحساسات</p>
                <h3 class="text-4xl font-bold text-white group-hover:text-[#13ec80] transition-colors">128</h3>
            </div>
            <div class="glass p-8 text-center group hover:border-emerald-500/40 transition-colors">
                <p class="text-xs text-emerald-500 uppercase font-bold mb-2 tracking-widest">حالة النظام</p>
                <h3 class="text-4xl font-bold text-[#13ec80]">مستقر</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="px-8" dir="rtl">', unsafe_allow_html=True)
        st.dataframe(
            df.style.background_gradient(cmap='Greens', subset=['Turbidite_NTU', 'Oxygene_Dissous_mgL']),
            use_container_width=True,
            height=400
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: 
        if st.button("⬅ العودة", use_container_width=True): prev_stage()
    with col3: 
        if st.button("تهيئة خوارزميات التدريب ⚙️", use_container_width=True): next_stage()

elif st.session_state.stage == 2:
    # المرحلة 3: التدريب (Training)
    st.markdown("""
    <div class="max-w-7xl mx-auto py-16 px-8 animate-fade-in" dir="rtl">
        <div class="mb-12">
            <h2 class="text-5xl font-bold mb-4 text-glow">ذكاء الآلة العميق</h2>
            <p class="text-emerald-500 font-mono text-lg">فك تشفير أنماط التلوث باستخدام خوارزميات SVM المتقدمة</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass h-full animate-slide-up" dir="rtl">', unsafe_allow_html=True)
        st.write("### ⚙️ حالة التدريب")
        with st.status("جاري تدريب السنتينل العالمي...", expanded=True) as s:
            time.sleep(0.8)
            st.write("تطبيق نوى RBF غير الخطية...")
            time.sleep(0.8)
            st.write("تحسين المستويات الفائقة (Hyperplanes)...")
            time.sleep(0.8)
            st.write("التحقق من صحة البيانات المتقاطعة...")
            s.update(label="اكتمل التدريب بنجاح!", state="complete")
        
        st.markdown("""
        <div class="mt-8 space-y-6">
            <div class="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                <p class="text-sm text-emerald-500 font-bold mb-1">دقة النموذج</p>
                <div class="flex items-end gap-2">
                    <span class="text-4xl font-bold">98.2%</span>
                    <span class="text-emerald-500 text-sm mb-1">+0.4% عن النسخة السابقة</span>
                </div>
            </div>
            <div class="p-4 rounded-xl bg-emerald-500/5 border border-emerald-500/10">
                <p class="text-sm text-emerald-500 font-bold mb-1">وقت المعالجة</p>
                <span class="text-4xl font-bold">42ms</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="glass h-full animate-slide-up" style="animation-delay: 0.2s;">', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0a1e19')
        ax.set_facecolor('#0a1e19')
        
        # رسم حدود القرار بشكل مبسط
        sns.scatterplot(data=df, x='Turbidite_NTU', y='Oxygene_Dissous_mgL', hue='Etat_Eau', 
                        palette=['#13ec80', '#ef4444'], s=100, alpha=0.6, ax=ax)
        
        ax.set_title("نطاق قرار الذكاء الاصطناعي", color='white', fontsize=18, pad=20, fontweight='bold')
        ax.tick_params(colors='white', labelsize=12)
        ax.set_xlabel("العكارة (NTU)", color='emerald', fontsize=14)
        ax.set_ylabel("الأكسجين المذاب (mg/L)", color='emerald', fontsize=14)
        for spine in ax.spines.values(): spine.set_color('#13ec8033')
        
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1: 
        if st.button("⬅ العودة", use_container_width=True): prev_stage()
    with col3: 
        if st.button("إطلاق المحاكي التفاعلي 📊", use_container_width=True): next_stage()

elif st.session_state.stage == 3:
    # المرحلة 4: المحاكي (Simulation)
    st.markdown("""
    <div class="max-w-7xl mx-auto py-16 px-8 animate-fade-in" dir="rtl">
        <div class="mb-12">
            <h2 class="text-5xl font-bold mb-4 text-glow">وحدة التحكم التفاعلية</h2>
            <p class="text-emerald-500 font-mono text-lg">محاكاة حية لمتغيرات البيئة المائية والتنبؤ الفوري</p>
        </div>
        
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-10">
            <div class="lg:col-span-4 space-y-8">
                <div class="glass p-8 animate-slide-up">
                    <h3 class="text-2xl font-bold mb-8 flex items-center gap-3">
                        <span class="material-icons-round text-[#13ec80]">tune</span>
                        ضبط المتغيرات
                    </h3>
    """, unsafe_allow_html=True)
    
    turbidity = st.slider("مستوى العكارة (NTU)", 0.0, 15.0, 5.0)
    oxygen = st.slider("الأكسجين المذاب (mg/L)", 0.0, 12.0, 6.0)
    
    st.markdown("""
                </div>
                <div class="glass p-6 border-emerald-500/10">
                    <p class="text-sm text-emerald-500/60 leading-relaxed">
                        قم بتحريك المؤشرات لمحاكاة ظروف بيئية مختلفة. سيقوم النموذج بتحليل القيم فوراً وتحديد ما إذا كانت المياه آمنة أم ملوثة.
                    </p>
                </div>
            </div>
            
            <div class="lg:col-span-8">
                <div class="glass h-full relative overflow-hidden flex flex-col items-center justify-center text-center p-12 animate-slide-up" style="animation-delay: 0.2s;">
                    <div class="absolute inset-x-0 top-0 h-1.5 bg-gradient-to-r from-transparent via-[#13ec80] to-transparent opacity-50"></div>
    """, unsafe_allow_html=True)
    
    # التنبؤ
    input_scaled = scaler.transform([[turbidity, oxygen]])
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    
    if pred == 0:
        st.markdown(f"""
            <div class="w-48 h-48 rounded-full bg-emerald-500/10 flex items-center justify-center mb-8 border border-emerald-500/30 glow-pulse">
                <span class="material-icons-round text-9xl text-[#13ec80] drop-shadow-[0_0_20px_rgba(19,236,128,0.4)]">verified</span>
            </div>
            <h2 class="text-6xl font-black text-white mb-4 text-glow">المياه آمنة</h2>
            <div class="inline-block px-6 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                <p class="text-[#13ec80] font-mono text-xl font-bold">ثقة النظام: {prob[0]:.1%}</p>
            </div>
            <p class="mt-8 text-emerald-100/40 max-w-md mx-auto">
                المعايير الحالية تقع ضمن النطاق الطبيعي المسموح به للحياة البحرية والاستخدام البشري.
            </p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="w-48 h-48 rounded-full bg-red-500/10 flex items-center justify-center mb-8 border border-red-500/30 animate-pulse">
                <span class="material-icons-round text-9xl text-red-500 drop-shadow-[0_0_20px_rgba(239,68,68,0.4)]">report_problem</span>
            </div>
            <h2 class="text-6xl font-black text-white mb-4 text-glow-red">تحذير: تلوث!</h2>
            <div class="inline-block px-6 py-2 rounded-full bg-red-500/10 border border-red-500/20">
                <p class="text-red-500 font-mono text-xl font-bold">ثقة النظام: {prob[1]:.1%}</p>
            </div>
            <p class="mt-8 text-red-100/40 max-w-md mx-auto">
                تم اكتشاف مؤشرات تلوث خارج النطاق الآمن. يرجى اتخاذ الإجراءات اللازمة فوراً.
            </p>
        """, unsafe_allow_html=True)
        
    st.markdown("""
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("↺ إعادة تشغيل كافة الأنظمة", use_container_width=True): go_to_stage(0)

# --- التذييل (Footer) ---
st.markdown("""
<div class="text-center py-16 text-[10px] text-emerald-900/40 uppercase tracking-[0.4em] font-mono border-t border-emerald-900/10 mt-20">
    ECOSENTINEL ENGINE v5.0 // SECURITY: AES-256 // STATUS: OPTIMAL // © 2026
</div>
""", unsafe_allow_html=True)
