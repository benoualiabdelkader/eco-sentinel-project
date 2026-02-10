# 🌊 السنتينل البيئي | EcoSentinel AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://eco-sentinel.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📝 وصف المشروع | Description du Projet

**السنتينل البيئي** هو نظام ذكاء اصطناعي متطور مصمم لمراقبة جودة المياه في النظم البيئية المائية. يستخدم النظام خوارزميات التعلم الآلي (SVM) لتحليل البيانات الواردة من الحساسات (مثل العكارة والأكسجين المذاب) والتنبؤ بحالة التلوث في الوقت الفعلي.

**EcoSentinel** est un système d'intelligence artificielle avancé conçu pour surveiller la qualité de l'eau dans les écosystèmes aquatiques. Le système utilise des algorithmes d'apprentissage automatique (SVM) pour analyser les données des capteurs (telles que la turbidité et l'oxygène dissous) et prédire l'état de pollution en temps réel.

---

## ✨ المميزات | Caractéristiques

- **واجهة مستخدم سينمائية:** تصميم عصري يعتمد على Glassmorphism و Tailwind CSS.
- **تحليل فوري:** تنبؤات لحظية باستخدام نموذج SVM RBF.
- **محاكي تفاعلي:** إمكانية تجربة سيناريوهات بيئية مختلفة عبر أشرطة التمرير.
- **دقة عالية:** دقة تصل إلى 98% في اكتشاف حالات التلوث.
- **خفيف الوزن:** تم تحسين الكود ليعمل بأقل استهلاك للموارد (Caching).

---

## 🚀 التشغيل | Installation & Utilisation

### 1. المتطلبات | Prérequis
- Python 3.11+
- pip

### 2. التثبيت | Installation
```bash
git clone https://github.com/benoualiabdelkader/eco-sentinel-project.git
cd eco-sentinel-project
pip install -r requirements.txt
```

### 3. التشغيل | Lancement
```bash
streamlit run eco_interface.py
```

---

## 🛠 التقنيات المستخدمة | Technologies

- **Frontend:** Streamlit, Tailwind CSS, Custom CSS (Glassmorphism).
- **Machine Learning:** Scikit-learn (SVM RBF), Pandas, Numpy.
- **Visualization:** Matplotlib, Seaborn.

---

## 📊 هيكلة المشروع | Structure du Projet

- `eco_interface.py`: الواجهة الرئيسية للتطبيق.
- `train_eco_models.py`: سكربت تدريب النماذج وتقييمها.
- `eco_sentinel_dataset.csv`: قاعدة البيانات البيئية.
- `requirements.txt`: المكتبات المطلوبة.

---

## 📄 الترخيص | Licence
هذا المشروع مرخص بموجب رخصة MIT.

---
**تم التطوير بواسطة Manus AI لتعزيز الاستدامة البيئية.**
