# Project Name

A data science project for [your problem statement, e.g. *"time series forecasting of XYZ"*].  
This repository contains data preprocessing, exploratory analysis, modeling, and an interactive Streamlit application for results visualization.

---

## 📂 Repository Structure

```
project-name/
├── README.md                   <- Project overview and usage
├── requirements.txt            <- Main Python dependencies
├── .gitignore                  <- Ignored files for git
├── data/                       <- (optional placeholder for local datasets)
│   └── maybe_leave_out/        <- Not tracked / excluded from repo
│
├── notebooks/                  <- Explanatory notebooks (EDA & baseline only)
│   ├── 0_preprocessing.ipynb
│   ├── 1_exploration.ipynb
│   ├── 2_time_series_decomp.ipynb
│   └── 3_naive_baseline_model.ipynb
│
├── src/ (PK: Option A)             <- Authoritative source code
│   ├── __init__.py
│   ├── data/                   <- include here, or move content to utils/? could also be a placeholder for actual data sets, which we agreed on to not share in the repo
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── Prophet.py
│   │   ├── ANN.py
│   │   ├── LightGBM.py
│   │   └── train_utils.py      <- leave here, or move to utils/ if data/ or visualization also use it?
│   └── visualization/          <- include here, or move content to utils/?
│       ├── __init__.py
│       └── plots.py
│
├── src/ (PK: Option B)
│   ├── __init__.py
│   ├── data/                   <- include here only the data sets
│   │   ├── __init__.py
│   │   └── data.csv
│   ├── models/
│   │   ├── __init__.py
│   │   ├── Prophet.py
│   │   ├── ANN.py
│   │   └── LightGBM.py
│   └── utils/                  <- can be removed if it has not content (see other comments)
│       ├── __init__.py
│       ├── data.py             <- load_data + preprocess, could include here or in data/
│       ├── plots.py            <- here or in visualization/
│       └── train_utils.py      <- include if data/ or visualization/ depend on it train_utils.py
│
├── src/ (PK: Option C, prefered by me, best practice if project grows)
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── preprocess.py
│   │   ├── raw/                 <- Raw datasets (CSV)
│   │   │   ├── dataset1.csv
│   │   │   └── dataset2.csv
│   │   └── processed/           <- Processed datasets (CSV or parquet)
│   │       ├── dataset1_clean.csv
│   │       └── dataset2_clean.csv
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ANN.py
│   │   ├── LightGBM.py
│   │   ├── Prophet.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── train_utils.py
│   └── visualization/
│       ├── __init__.py
│       └── plots.py
│
├── app/                        <- Interactive Streamlit application
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── requirements.txt        <- App-specific dependencies
│   ├── data/
│   │   ├── __init__.py
│   │   └── data.csv
│   ├── pages/
│   │   ├── __init__.py
│   │   └── page1.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_processing.py
│   │   └── visualization.py
│   └── images/
│       ├── __init__.py
│       └── logo.png
│
├── reports/                    <- Generated outputs and presentation
│   ├── figures/
│   │   └── figure1.png
│   └── Presentation.pptx
│
└── configs/                    <- Configuration files
    └── config.yaml
```

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Workflow

### Data
- Place raw datasets into `data/` (not tracked in GitHub).  
- Processed data is generated using `src/data/preprocess.py`.  

### Notebooks (storytelling only)
- **0_preprocessing.ipynb** → demonstrates data cleaning steps.  
- **1_exploration.ipynb** → exploratory analysis and visualization.  
- **2_time_series_decomp.ipynb** → seasonal/trend decomposition.  
- **3_naive_baseline_model.ipynb** → simple baseline for benchmarking.  

### Models
- Located in `src/models/`  
  - `Prophet.ipynb` – Prophet model implementation  
  - `ANN.ipynb` – Neural network model  
  - `LightGBM.ipynb` – Gradient boosting model  
  - `train_utils.py` – shared functions for training/evaluation  

### Visualizations
- Custom plots in `src/visualization/plots.py`.  
- Figures stored in `reports/figures/`.  

---

## 🚀 Running the Streamlit App

Navigate to the `app/` directory and install its dependencies:

```bash
cd app
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run streamlit_app.py
```

This launches an interactive dashboard with multiple pages, powered by data in `app/data/`.

---

## 📑 Reports

- **Figures** → `reports/figures/`  
- **Presentation slides** → `reports/Presentation.pptx`  

---

## 📌 Notes

- `src/` contains the authoritative code for reproducibility.  
- `notebooks/` are explanatory and showcase data preparation, EDA, and baseline results.  
- Use `configs/config.yaml` for project settings (paths, parameters).  
- Raw data should **not** be committed to GitHub.  

---

## ✨ Authors

This project was developed collaboratively by the entire project team.  
**All members contributed equally to every stage of the project, including data preparation, modeling, visualization, app development, and reporting.**

- [Your Name]  
- [Collaborator 1]  
- [Collaborator 2]