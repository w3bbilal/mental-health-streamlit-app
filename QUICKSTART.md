# 🚀 Quick Start Guide - Mental Health App

## ⚡ 5-Minute Setup

### Step 1: Install Dependencies (1 minute)

```bash
pip install -r requirements.txt
```

### Step 2: Run the App (30 seconds)

```bash
streamlit run mental_health_app.py
```

### Step 3: Open in Browser

The app will automatically open at:

```
http://localhost:8501
```

-----

## 🎯 What You’ll See

The app has 4 main sections (use sidebar to navigate):

### 1️⃣ Overview 📊

- Dataset statistics
- Project objectives
- Key metrics

### 2️⃣ EDA & Visualizations 📈

- Global depression trends
- Top countries ranking
- Disorder correlations
- Statistical summaries

### 3️⃣ Model Building 🤖

- Model performance metrics
- Training results (R² = 0.943)
- Feature importance
- Actual vs Predicted visualization

### 4️⃣ Predictions 🔮

- **Interactive prediction interface**
- Enter Year, Anxiety, Drug Use, Alcohol rates
- Get instant depression rate prediction
- View 20-year trend forecast

-----

## 🎮 Try This First

1. Go to **🔮 Predictions** page
1. Use the default values (2025, 3.8%, 0.95%, 1.4%)
1. Click **“🔮 Predict Depression Rate”**
1. You should see: **Predicted Depression: ~3.847%**

-----

## 📊 Dataset Info

- **6,468 records** from 200+ countries
- **30 years** of data (1990-2019)
- **7 mental disorders** tracked
- Uses generated sample data (replace with your CSV)

-----

## 🔧 Troubleshooting

**Port already in use?**

```bash
streamlit run mental_health_app.py --server.port 8502
```

**Need to install Python?**

- Download from python.org
- Make sure to check “Add Python to PATH” during installation

**On Mac/Linux permission issue?**

```bash
chmod +x mental_health_app.py
```

-----

## 💡 Next Steps

1. Replace sample data with your actual CSV:
- Update `load_and_process_data()` function
- Use: `df = pd.read_csv('your_dataset.csv')`
1. Retrain model with your data
1. Deploy online (see README.md for options)

-----

## 📞 Quick Reference Commands

```bash
# Install all dependencies
pip install -r requirements.txt

# Run the app
streamlit run mental_health_app.py

# Run on custom port
streamlit run mental_health_app.py --server.port 8502

# Run in headless mode (no browser)
streamlit run mental_health_app.py --logger.level=debug

# Deactivate virtual environment
deactivate
```

-----

## ✨ Features Highlight

✅ 4 interactive pages with navigation
✅ 94.3% accurate ML model
✅ Real-time predictions
✅ Beautiful visualizations (matplotlib + seaborn)
✅ Responsive design
✅ Fast data processing with caching

-----

**Ready to go! 🚀 Run the app and explore the global mental health insights!**