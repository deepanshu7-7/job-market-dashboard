# 📊 Job Market Intelligence Dashboard

> An ML-powered dashboard analysing **12,000 real DS/ML job postings** from Naukri.com to help aspiring data scientists understand the Indian job market — and where they stand in it.

🚀 **[Live Demo →](https://job-market-dashboard-cjyrh5y2u4ban5babyigct.streamlit.app/)**  &nbsp;|&nbsp;  📁 **[Dataset — Naukri.com India](https://www.kaggle.com/datasets/anandhuh/data-science-jobs-in-india)**

---

## 📸 Dashboard Preview

> *Add a screenshot here — take a screenshot of your running dashboard and drag it into this README on GitHub*

---

## ✨ Features

| Tab | What it does |
|-----|-------------|
| 📊 **Market Overview** | Skill demand bar chart, city distribution, top hiring companies, role ranking bubble chart |
| 🤖 **Experience Predictor** | Pick a role + city + your skills → ML model predicts experience required |
| 🔍 **Skill Gap Analyser** | Enter your skills → see match % for every role + exactly what to learn next |
| 🗂 **Raw Data Explorer** | Search and filter all 12,000 job postings, download as CSV |

---

## 🧠 Machine Learning Model

| Property | Value |
|----------|-------|
| Algorithm | Random Forest Regressor |
| Trees | 300 |
| Max Depth | 15 |
| Features | 19 |
| Training rows | 12,000 |
| Test R² Score | **99.9%** |

**Features used for prediction:**
- Job role (encoded)
- City (encoded)
- Min / Max experience range
- 14 skill flags — Python, SQL, ML, Deep Learning, NLP, Tableau, Power BI, AWS, Azure, Spark, TensorFlow, Statistics, Excel, Docker

---

## 🛠 Tech Stack

```
Python 3.x
Pandas          — data cleaning and manipulation
Scikit-learn    — Random Forest model + LabelEncoder
Plotly          — interactive charts
Streamlit       — web dashboard
Pickle          — saving and loading trained model
```

---

## 📁 Project Structure

```
job-market-dashboard/
│
├── app.py                              ← Streamlit dashboard (main file)
├── model.pkl                           ← trained Random Forest model
├── encoders.pkl                        ← LabelEncoders + skill list
├── naukri_data_science_jobs_india.csv  ← dataset (12,000 rows)
├── requirements.txt                    ← Python dependencies
└── README.md                           ← you are here
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/deepanshu7-7/job-market-dashboard.git
cd job-market-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the dashboard**
```bash
streamlit run app.py
```

Browser opens automatically at `http://localhost:8501`

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly==5.18.0
scikit-learn
```

---

## 📊 Key Insights from the Data

- **Python** is the most in-demand skill — appears in 60%+ of all DS/ML job postings
- **SQL** is second — essential alongside Python for almost every data role
- **Bangalore/Bengaluru** accounts for the majority of DS/ML jobs in India
- **Data Engineer** roles have the highest number of openings
- **Data Architect** and **Lead Data Scientist** require the most experience (8+ years)
- Entry-level roles like **Data Analyst** typically need 2–4 years experience

---

## 🔄 How the ML Pipeline Works

```
Raw CSV (12,000 rows)
        ↓
Data Cleaning
(standardise text, parse experience "3-6" → Min=3, Max=6, Avg=4.5)
        ↓
Feature Engineering
(14 skill flags from job descriptions, LabelEncode city + role)
        ↓
Train Random Forest (300 trees, max_depth=15)
        ↓
Save model.pkl + encoders.pkl
        ↓
Streamlit loads pkl → user inputs → predict() → result
```

---

## 👨‍💻 Author

**Deepanshu Kumar**  
BCA Graduate · Amity University, Punjab  
Aspiring Data Scientist

📧 deepanshukumar3232@gmail.com  
🐙 [github.com/deepanshu7-7](https://github.com/deepanshu7-7)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with Python · Pandas · Scikit-learn · Plotly · Streamlit*  
*Data source: Naukri.com India via Kaggle*
