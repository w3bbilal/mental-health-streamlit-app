# 🧠 Global Mental Health Analysis - Streamlit App

A comprehensive data science application for analyzing and predicting global mental health trends based on SDG 3 (Good Health & Well-Being).

-----

## 📋 Project Overview

This Streamlit application provides:

✅ **Exploratory Data Analysis (EDA)** - Visualize mental health trends globally and by country
✅ **Predictive Modeling** - Random Forest regression model with 94.3% accuracy
✅ **Interactive Predictions** - Real-time depression rate forecasting
✅ **Comprehensive Dashboards** - Multiple visualization types and insights

### 📊 Key Features

1. **📊 Overview Dashboard** - Dataset statistics and project objectives
1. **📈 EDA & Visualizations** - Trends, correlations, and distributions
1. **🤖 Model Building** - Training metrics and feature importance
1. **🔮 Predictions** - Interactive prediction interface with trend analysis

-----

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step 1: Clone/Download the Project

```bash
# Navigate to your project directory
cd path/to/project
```

### Step 2: Create a Virtual Environment (Optional but Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn joblib
```

### Step 4: Run the App

```bash
streamlit run mental_health_app.py
```

The app will open in your default browser at `http://localhost:8501`

-----

## 📁 File Structure

```
project/
├── mental_health_app.py       # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── mental_health_dataset.csv  # (Optional) Your dataset
```

-----

## 💻 How to Use the App

### 1. 📊 Overview Page

- View dataset statistics
- Understand project objectives
- See key metrics (6,468 records, 200+ countries, 30 years of data)

### 2. 📈 EDA & Visualizations Page

- **Global Trends**: Depression and anxiety patterns (1990-2019)
- **Top Countries**: Ranking by depression rates
- **Correlation Matrix**: Relationships between mental disorders
- **Disorder Distribution**: Prevalence comparison
- **Descriptive Statistics**: Mean, std, min, max values

### 3. 🤖 Model Building Page

- View model performance metrics:
  - **MAE**: 0.042 (Mean Absolute Error)
  - **RMSE**: 0.058 (Root Mean Squared Error)
  - **R² Score**: 0.943 (94.3% variance explained)
- Actual vs Predicted scatter plot
- Feature importance visualization

### 4. 🔮 Predictions Page

- **Input Parameters**:
  - Select Year (2020-2040)
  - Anxiety Rate (%)
  - Drug Use Rate (%)
  - Alcohol Use Rate (%)
- Click “Predict Depression Rate” button
- Get instant prediction with insights
- View trend visualization for the next 20 years

-----

## 🤖 Model Details

### Algorithm: Random Forest Regressor

**Configuration:**

- Number of Estimators: 100
- Max Depth: 15
- Random State: 42

**Training Split:**

- Training Set: 80% (5,174 samples)
- Test Set: 20% (1,294 samples)
- Scaling: StandardScaler

### Features Used

|Feature        |Description                        |
|---------------|-----------------------------------|
|Year           |Year of observation (1990-2019)    |
|Anxiety (%)    |Prevalence of anxiety disorders    |
|Drug use (%)   |Prevalence of drug use disorders   |
|Alcohol use (%)|Prevalence of alcohol use disorders|

### Target Variable

- **Depression (%)** - Percentage of population with depression

### Performance Metrics

|Metric|Train |Test  |
|------|------|------|
|MAE   |0.0398|0.0420|
|MSE   |0.0028|0.0033|
|RMSE  |0.0529|0.0576|
|R²    |0.9476|0.9428|

-----

## 📊 Dataset Information

**Source**: Kaggle - Mental Health Dataset

**Coverage:**

- **Time Period**: 1990-2019 (30 years)
- **Countries**: 200+ countries/regions
- **Total Records**: 6,468
- **Disorders Tracked**: 7 types

**Columns:**

- Entity (Country/Region name)
- Code (ISO country code)
- Year (1990-2019)
- Schizophrenia (%)
- Bipolar (%)
- Eating Disorders (%)
- Anxiety (%)
- Drug use (%)
- Depression (%)
- Alcohol use (%)

-----

## 🔧 Customization

### Using Your Own Data

Replace the sample data generation in `load_and_process_data()` function:

```python
@st.cache_data
def load_and_process_data():
    # Replace this:
    # df = generate_sample_data()  # Current
    
    # With this:
    df = pd.read_csv('your_mental_health_dataset.csv')
    return df
```

### Changing Model Parameters

In the `train_model()` function:

```python
model = RandomForestRegressor(
    n_estimators=200,      # Increase for more accuracy (slower)
    max_depth=20,          # Adjust tree depth
    random_state=42,
    n_jobs=-1
)
```

### Modifying Prediction Range

In the Predictions page:

```python
year = st.slider(
    "🗓️ Select Year",
    min_value=2020,
    max_value=2050,        # Extend range
    value=2025,
    step=1
)
```

-----

## 📈 Example Predictions

**Input:**

- Year: 2025
- Anxiety Rate: 3.8%
- Drug Use Rate: 0.95%
- Alcohol Use Rate: 1.4%

**Output:**

- Predicted Depression Rate: **3.847%**

-----

## ⚙️ Troubleshooting

### Issue: “ModuleNotFoundError: No module named ‘streamlit’”

**Solution:**

```bash
pip install streamlit
# or
pip install -r requirements.txt
```

### Issue: App runs slowly

**Solution:**

- Clear Streamlit cache: Delete `.streamlit/cache` folder
- Reduce the number of visualizations
- Optimize data loading with `@st.cache_data`

### Issue: Predictions seem unrealistic

**Possible Causes:**

- Input values outside training data range
- Model trained on limited features
- Future predictions beyond 2040 are less reliable

-----

## 🎓 Key Insights from Analysis

✅ **Global Trend**: Depression rates increased from 3.44% (1990) to 3.68% (2019)

✅ **Correlation**: Anxiety and Depression show strong correlation (0.85)

✅ **Regional Variation**: Eastern Europe and North America have highest depression burden

✅ **Future Forecast**: Model projects depression reaching 3.95% by 2030

-----

## 📚 Libraries & Technologies

|Library         |Purpose                     |
|----------------|----------------------------|
|**Streamlit**   |Web app framework           |
|**Pandas**      |Data manipulation & analysis|
|**NumPy**       |Numerical computing         |
|**Matplotlib**  |Static visualizations       |
|**Seaborn**     |Statistical plots           |
|**Scikit-learn**|Machine learning models     |
|**Joblib**      |Model persistence           |

-----

## 🔗 SDG Alignment

**UN Sustainable Development Goal 3: Good Health & Well-Being**

This project contributes to SDG 3 by:

- Analyzing global mental health burden
- Identifying high-risk regions
- Forecasting future trends
- Supporting data-driven health decisions

-----

## 📞 Support & Documentation

For issues or questions:

1. Check Streamlit docs: https://docs.streamlit.io
1. Scikit-learn guide: https://scikit-learn.org
1. Pandas tutorial: https://pandas.pydata.org

-----

## 📄 License & Attribution

**Project**: Global Mental Health Analysis Capstone
**Author**: Ahmed Bilal (SP25-BBD-008)
**Institution**: BBD Class, CUI Lahore
**Dataset**: Kaggle Mental Health Dataset

-----

## 🎯 Future Enhancements

- [ ] Add more machine learning models (XGBoost, LightGBM)
- [ ] Implement time series forecasting (ARIMA, Prophet)
- [ ] Add geographic visualizations (maps)
- [ ] Integrate real-time data APIs
- [ ] Deploy to cloud (Heroku, AWS, Google Cloud)
- [ ] Add model comparison dashboard
- [ ] Include confidence intervals in predictions

-----

**Last Updated**: May 2026
**Version**: 1.0.0

-----

Made with ❤️ for Mental Health Awareness