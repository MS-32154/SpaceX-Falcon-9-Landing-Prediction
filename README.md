# SpaceX Falcon 9 Landing Prediction

A comprehensive data science project that predicts SpaceX Falcon 9 first stage landing success using machine learning, enabling competitive launch cost estimation.

## 🚀 Project Overview

SpaceX advertises Falcon 9 launches at $62 million, significantly less than competitors who charge upward of $165 million. This cost advantage stems from SpaceX's ability to reuse the first stage. By predicting landing success, we can estimate launch costs and provide valuable competitive intelligence for alternate launch providers.

## 📊 Project Workflow

### 1. Data Collection (`001_data_collection.ipynb`)
- **SpaceX API**: Collected structured launch data from official SpaceX API
- **Web Scraping**: Extracted historical records from Wikipedia for validation
- **Dataset**: 90+ Falcon 9 launches with 16 features including payload mass, orbit type, launch site, and booster characteristics

### 2. SQL Analysis (`002_sql_analysis.ipynb`)
- Comprehensive SQL queries to explore launch patterns
- Analyzed success rates by launch site, orbit type, and booster version
- Identified temporal trends showing improvement from 40% to 90%+ success rate
- Examined booster reusability metrics and hardware configurations

### 3. Data Wrangling (`003_data_wrangling.ipynb`)
- Created binary classification labels (Success/Failure)
- Handled missing values and outliers
- Feature analysis revealing Block 5 boosters achieve highest success rates
- Validated that reused boosters perform comparably to new ones

### 4. Exploratory Data Analysis (`004_data_visualization.ipynb`)
- Visual pattern discovery using matplotlib and seaborn
- Key findings:
  - Success rate improves with flight experience
  - LEO/ISS missions show 80-100% success rates
  - GTO missions remain challenging (~60%) due to high energy requirements
  - Landing hardware (GridFins + Legs) critical for success

### 5. Geographic Analysis (`005_launch_site_location.ipynb`)
- Interactive Folium maps of launch site locations
- Proximity analysis to coastlines, railways, highways, and cities
- All sites strategically positioned:
  - <1.5 km from coast (safety and recovery)
  - 20+ km from major cities (safety buffer)
  - Excellent rail/highway access (logistics)

### 6. Interactive Dashboard (`006_spacex-dash-app.py`)
- **Dash/Plotly** web application for real-time data exploration
- Features:
  - Launch site filtering
  - Payload range selection
  - Success rate visualizations
  - Booster version distribution
  - Summary statistics tables

**Run Dashboard:**
```bash
python 006_spacex-dash-app.py
```
Access at `http://localhost:8050`

### 7. Machine Learning Models (`007_ml_prediction.ipynb`)
Trained and evaluated 7 classification algorithms:
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

**Performance Highlights:**
- Best Model: **XGBoost**
- Accuracy: **94.1%**
- Precision: **93.3%**
- F1-Score: **96.6%**

## 🎯 Key Findings

### Technical Insights
1. **Experience Matters**: Success rate correlates strongly with flight number (+0.5 correlation)
2. **Technology Evolution**: Block 5 boosters show 90%+ success rates
3. **Hardware Critical**: GridFins and landing legs essential for success
4. **Reusability Works**: Reused boosters perform as well as new ones

### Business Impact
- **SpaceX Average Cost**: $68 million (with 75% landing success)
- **Competitor Cost**: $165 million
- **Cost Savings**: $97 million per launch (59% advantage)
- **Prediction Accuracy**: 94% enables reliable cost estimation

## 🛠️ Technologies Used

**Languages & Libraries:**
- Python 3.13
- pandas, numpy (data manipulation)
- matplotlib, seaborn, plotly (visualization)
- folium (geographic mapping)
- scikit-learn (machine learning)
- XGBoost (advanced modeling)
- Dash (web dashboard)
- BeautifulSoup (web scraping)

**Tools:**
- Jupyter Notebook
- SQLite
- Git/GitHub

## 📁 Project Structure

```
spacex-landing-prediction/
│
├── 001_data_collection.ipynb       # API & web scraping
├── 002_sql_analysis.ipynb          # SQL exploratory analysis
├── 003_data_wrangling.ipynb        # Data cleaning & labeling
├── 004_data_visualization.ipynb    # EDA with visualizations
├── 005_launch_site_location.ipynb  # Geographic analysis
├── 006_spacex-dash-app.py          # Interactive dashboard
├── 007_ml_prediction.ipynb         # Machine learning models
│
├── spacex_falcon9_dataset.csv      # Primary dataset
├── spacex_falcon9_labeled.csv      # Labeled dataset
├── spacex_falcon9_features.csv     # Engineered features
├── spacex_launch_dash_wiki.csv     # Dashboard data
└── README.md
```

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn plotly dash folium scikit-learn xgboost beautifulsoup4 requests imbalanced-learn
```

### Run the Analysis
1. **Data Collection**: Execute `001_data_collection.ipynb`
2. **SQL Analysis**: Execute `002_sql_analysis.ipynb`
3. **Data Wrangling**: Execute `003_data_wrangling.ipynb`
4. **Visualization**: Execute `004_data_visualization.ipynb`
5. **Geographic Analysis**: Execute `005_launch_site_location.ipynb`
6. **Dashboard**: Run `python 006_spacex-dash-app.py`
7. **ML Models**: Execute `007_ml_prediction.ipynb`

## 📈 Results Summary

| Model               | Accuracy | Precision | Recall   | F1-Score | ROC-AUC  |
| ------------------- | -------- | --------- | -------- | -------- | -------- |
| XGBoost             | 0.941176 | 0.933333  | 1.000000 | 0.965517 | 0.982143 |
| Gradient Boosting   | 0.911765 | 0.962963  | 0.928571 | 0.945455 | 0.937500 |
| Decision Tree       | 0.882353 | 0.928571  | 0.928571 | 0.928571 | 0.723214 |
| Random Forest       | 0.882353 | 0.961538  | 0.892857 | 0.925926 | 0.886905 |
| Logistic Regression | 0.794118 | 0.862069  | 0.892857 | 0.877193 | 0.857143 |
| KNN                 | 0.794118 | 0.888889  | 0.857143 | 0.872727 | 0.678571 |
| SVM                 | 0.794118 | 0.920000  | 0.821429 | 0.867925 | 0.863095 |


## 🎓 Skills Demonstrated

- Data collection via APIs and web scraping
- SQL database querying and analysis
- Data cleaning and feature engineering
- Exploratory data analysis with visualizations
- Geographic data analysis and mapping
- Interactive dashboard development
- Machine learning classification
- Model evaluation and comparison
- Business impact analysis

## 📝 License

This project is part of an applied data science capstone and is available for educational purposes.

## 🙏 Acknowledgments

- SpaceX API for launch data
- Wikipedia for historical launch records
- IBM Data Science Professional Certificate
- Coursera

---
 
**Date**: November 2025
