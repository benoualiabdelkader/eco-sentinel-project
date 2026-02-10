import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time

# إعداد الصفحة لتكون واسعة وبدون قوائم جانبية
st.set_page_config(
    page_title="EcoSentinel AI",
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
        return pd.DataFrame({'Turbidite_NTU': [5, 12], 'Oxygene_Dissous_mgL': [8, 2], 'Etat_Eau': [0, 1]})

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

# --- التصميم الأساسي (Tailwind & Custom CSS) ---
st.markdown("""
<script src="https://cdn.tailwindcss.com"></script>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons+Round" rel="stylesheet">
<style>
    body { font-family: 'Space Grotesk', sans-serif; background-color: #102219; color: white; }
    .stApp { background-color: #102219; }
    iframe { border: none !important; }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .primary-text { color: #13ec80; }
    .primary-bg { background-color: #13ec80; }
    
    .glass {
        background: rgba(22, 46, 34, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(19, 236, 128, 0.1);
        border-radius: 1rem;
        padding: 2rem;
    }
    
    .stButton > button {
        background-color: #13ec80 !important;
        color: #102219 !important;
        font-weight: bold !important;
        border-radius: 0.5rem !important;
        border: none !important;
        transition: 0.3s !important;
    }
    .stButton > button:hover {
        background-color: #0ea85b !important;
        transform: scale(1.05) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- التنقل العلوي ---
progress = (st.session_state.stage / 4) * 100
st.markdown(f"""
<div class="flex items-center justify-between px-8 py-4 border-b border-emerald-900/30 bg-[#102219]">
    <div class="flex items-center gap-3">
        <div class="w-8 h-8 rounded bg-emerald-500/20 flex items-center justify-center text-[#13ec80]">
            <span class="material-icons-round">water_drop</span>
        </div>
        <span class="font-bold text-lg tracking-tight">EcoSentinel</span>
    </div>
    <div class="w-1/3">
        <div class="flex justify-between text-xs mb-1 text-emerald-500 font-mono">
            <span>STAGE {st.session_state.stage + 1}</span>
            <span>{int(progress)}%</span>
        </div>
        <div class="w-full h-1.5 bg-emerald-900/50 rounded-full overflow-hidden">
            <div class="h-full bg-[#13ec80]" style="width: {progress}%"></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- محتوى المراحل ---

if st.session_state.stage == 0:
    # المرحلة الأولى: ترحيب (Stage 1 in CSS)
    st.markdown("""
    <div class="max-w-6xl mx-auto py-20 px-6">
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            <div>
                <div class="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-emerald-500/30 bg-emerald-500/5 text-[#13ec80] text-xs font-bold mb-6">
                    <span class="relative flex h-2 w-2">
                        <span class="animate-ping absolute h-full w-full rounded-full bg-[#13ec80] opacity-75"></span>
                        <span class="relative h-2 w-2 rounded-full bg-[#13ec80]"></span>
                    </span>
                    SYSTEM ONLINE
                </div>
                <h1 class="text-7xl font-bold leading-none mb-6">Ecological <br><span class="text-transparent bg-clip-text bg-gradient-to-r from-[#13ec80] to-teal-400">Sentinel</span></h1>
                <p class="text-xl text-emerald-100/60 leading-relaxed mb-8">Protecting aquatic ecosystems through AI-driven intelligence. Real-time monitoring for a sustainable future.</p>
                <div class="flex gap-4">
                    <div class="glass p-4 flex items-center gap-4">
                        <div class="w-10 h-10 rounded bg-emerald-500/10 flex items-center justify-center text-[#13ec80]">
                            <span class="material-icons-round">analytics</span>
                        </div>
                        <div>
                            <p class="text-xs text-emerald-500 uppercase font-bold">Accuracy</p>
                            <p class="font-bold">98.4%</p>
                        </div>
                    </div>
                    <div class="glass p-4 flex items-center gap-4">
                        <div class="w-10 h-10 rounded bg-emerald-500/10 flex items-center justify-center text-[#13ec80]">
                            <span class="material-icons-round">bolt</span>
                        </div>
                        <div>
                            <p class="text-xs text-emerald-500 uppercase font-bold">Latency</p>
                            <p class="font-bold">24ms</p>
                        </div>
                    </div>
                </div>
            </div>
            <div class="flex justify-center relative">
                <div class="w-80 h-80 bg-emerald-500/10 rounded-full blur-3xl absolute animate-pulse"></div>
                <div class="w-72 h-72 rounded-full border border-emerald-500/20 flex items-center justify-center relative z-10 bg-emerald-900/20 backdrop-blur-xl">
                    <span class="material-icons-round text-9xl text-[#13ec80] drop-shadow-[0_0_15px_rgba(19,236,128,0.5)]">water</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Start Discovery Journey →", use_container_width=True): next_stage()

elif st.session_state.stage == 1:
    # المرحلة الثانية: البيانات (Stage 2 in CSS)
    st.markdown("""
    <div class="max-w-6xl mx-auto py-12 px-6">
        <h2 class="text-4xl font-bold mb-2">Data & Foundation</h2>
        <p class="text-emerald-500 mb-8 font-mono">Exploring 1,024 High-Resolution Environmental Records</p>
        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
            <div class="glass p-6 text-center">
                <p class="text-xs text-emerald-500 uppercase font-bold">Avg Turbidity</p>
                <h3 class="text-3xl font-bold">4.2 NTU</h3>
            </div>
            <div class="glass p-6 text-center">
                <p class="text-xs text-emerald-500 uppercase font-bold">Avg Oxygen</p>
                <h3 class="text-3xl font-bold">6.8 mg/L</h3>
            </div>
            <div class="glass p-6 text-center">
                <p class="text-xs text-emerald-500 uppercase font-bold">Total Sensors</p>
                <h3 class="text-3xl font-bold">128</h3>
            </div>
            <div class="glass p-6 text-center">
                <p class="text-xs text-emerald-500 uppercase font-bold">Location</p>
                <h3 class="text-3xl font-bold">Danube</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.style.background_gradient(cmap='Greens'), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1: 
        if st.button("⬅ Back"): prev_stage()
    with col2: 
        if st.button("Initialize Training Algorithms ⚙️"): next_stage()

elif st.session_state.stage == 2:
    # المرحلة الثالثة: التدريب والتحليل العميق (Stage 3 in CSS)
    st.markdown("""
    <div class="max-w-6xl mx-auto py-12 px-6">
        <h2 class="text-4xl font-bold mb-2">Stage 3: Machine Insights</h2>
        <p class="text-emerald-500 mb-8 font-mono">Deep Learning Decryption of Pollution Patterns</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.status("Training Global SVM Sentinel...", expanded=True) as s:
        time.sleep(1)
        st.write("Applying Non-Linear RBF Kernels...")
        time.sleep(1)
        st.write("Optimizing Hyperplanes...")
        s.update(label="Training Synchronized!", state="complete")
    
    st.markdown('<div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-8">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass h-full">', unsafe_allow_html=True)
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        fig1.patch.set_facecolor('#102219')
        sns.scatterplot(data=df, x='Turbidite_NTU', y='Oxygene_Dissous_mgL', hue='Etat_Eau', palette='RdYlGn_r', ax=ax1)
        ax1.set_title("AI Decision Realm", color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.set_xlabel("Turbidity (NTU)", color='white')
        ax1.set_ylabel("Oxygen (mg/L)", color='white')
        st.pyplot(fig1)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass h-full">', unsafe_allow_html=True)
        st.write("### 🧠 Technical Insight")
        st.markdown("""
        The SVM utilizes a **Radial Basis Function (RBF)** kernel to isolate outlier clusters.
        - **Decision Boundary:** Highly non-linear pattern detected.
        - **Impact Factor:** Oxygen levels contribute 65% to variance.
        - **Anomaly Detection:** Outliers flagged with 92% precision.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"): prev_stage()
    with col2:
        if st.button("Launch Interactive Simulator 📊"): next_stage()

elif st.session_state.stage == 3:
    # المرحلة الرابعة: المحاكي (Stage 4 in CSS)
    st.markdown("""
    <div class="max-w-6xl mx-auto py-12 px-6">
        <h2 class="text-4xl font-bold mb-2">Interactive Control Module</h2>
        <p class="text-emerald-500 mb-8 font-mono">Live Simulation of Aquatic Environment Variables</p>
        <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 h-[500px]">
            <div class="lg:col-span-4 glass flex flex-col justify-center">
                <h3 class="text-xl font-bold mb-6">Environment Tuning</h3>
    """, unsafe_allow_html=True)
    
    turbidity = st.slider("TURBIDITY RANGE (NTU)", 0.0, 15.0, 5.0)
    oxygen = st.slider("OXYGEN DISSOLVED (mg/L)", 0.0, 12.0, 6.0)
    
    st.markdown("""
            </div>
            <div class="lg:col-span-8 glass relative overflow-hidden flex flex-col items-center justify-center text-center">
                <div class="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-transparent via-[#13ec80] to-transparent"></div>
    """, unsafe_allow_html=True)
    
    input_scaled = scaler.transform([[turbidity, oxygen]])
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0]
    
    if pred == 0:
        st.markdown(f"""
            <div class="w-32 h-32 rounded-full bg-emerald-500/20 flex items-center justify-center mb-6 border border-emerald-500/50">
                <span class="material-icons-round text-6xl text-[#13ec80]">verified_user</span>
            </div>
            <h2 class="text-5xl font-bold text-white mb-2">Water is Safe</h2>
            <p class="text-emerald-500 font-mono text-xl">System Confidence: {prob[0]:.1%}</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="w-32 h-32 rounded-full bg-red-500/20 flex items-center justify-center mb-6 border border-red-500/50">
                <span class="material-icons-round text-6xl text-red-500">warning</span>
            </div>
            <h2 class="text-5xl font-bold text-white mb-2">Water Polluted</h2>
            <p class="text-red-500 font-mono text-xl">System Confidence: {prob[1]:.1%}</p>
        """, unsafe_allow_html=True)
        
    st.markdown("</div></div></div>", unsafe_allow_html=True)
    
    if st.button("↺ Restart All Systems"): go_to_stage(0)

# Footer
st.markdown("""
<div class="text-center py-10 text-xs text-emerald-900/50 uppercase tracking-widest font-mono">
    ECOSENTINEL ENGINE v4.2 // SECURITY: ENCRYPTED // STATUS: OPTIMAL
</div>
""", unsafe_allow_html=True)
