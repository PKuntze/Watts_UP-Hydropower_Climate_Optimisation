# 🌊 Hydropower Forecasting with Machine Learning
IBM SkillsBuild Hydropower Climate Optimisation Challenge [Zindi Africa](https://zindi.africa/competitions/ibm-skillsbuild-hydropower-climate-optimisation-challenge)

This document outlines our team’s approach to improving hydropower generation forecasts for off-grid communities using climate data and machine learning. Reliable hydropower predictions can enhance local energy planning and guide infrastructure optimization.

We compared several time-series forecasting methods, evaluating their RMSE against benchmark submissions from other participants in the challenge. The three most promising models were:

- **Multilayer Perceptron (MLP)** — a deep neural network with dropout layers and Monte Carlo uncertainty estimation.
- **LightGBM** 
- **Prophet** — a time-series model developed by Meta, well-suited for capturing seasonality and long-term trends.

The MLP achieved the best performance with a Private Leaderboard **RMSE of 4.24**, which would have ranked first if the challenge were still open.

This repository allows users to reproduce the training pipeline and explore forecasts via a Streamlit dashboard. It includes data preprocessing, exploratory analysis, modeling, and evaluation.

🔑 **Key Elements**:

- Feature Engineering: Lagged, rolling, and climate-based features to represent temporal and environmental dependencies.
- Recursive Prediction: Multi-step forecasting by iteratively feeding model outputs back as new inputs.
- Uncertainty Estimation: Monte Carlo Dropout in the MLP to generate confidence intervals for each prediction.
- Model Comparison: Evaluation of MLP, LightGBM, and Prophet using RMSE on training, testing, and unseen (extra month) datasets.



---

## 📂 Repository Structure
<details>
  <summary>Structure Visualization</summary>
    
  ```
project-name/
├── README.md                   <- Project overview and usage
├── requirements.txt            <- Main Python dependencies
├── .gitignore                  <- Ignored files for git
│
├── notebooks/                  <- Explanatory notebooks (EDA & baseline only)
│   ├── 01_data_preparation_and_exploration.ipynb
│   ├── 02_naive_baseline_model.ipynb
│   ├── 03_LightGBM_model_test.ipynb
│   ├── 04_Prophet_model_test.ipynb
│   ├── 05_ANN_model_test.ipynb
│   └── 06_XGBoost_model_test.ipynb
│
├── src/ 
│   ├── data/                   <- Not shared publicly
│   ├── model/
│   │   └── ANN.py              <- Winning model
│   ├── plots/
│   │   └── ANN_median_pc_niter1000_batch500_1/    <- Folder dynamically created by ANN.py
│   │       ├── feature_importance.png
│   │       ...                                    <- Various plots for validation and prediction
│   │       └── residuals_scatter_Train.png
│   └── submissions             <- Not shared publicly
│
├── app/                        <- Interactive Streamlit application
│   ├── __init__.py
│   ├── streamlit_app.py
│   ├── requirements.txt        <- App-specific dependencies
│   ├── .streamlit/
│   │   └── config.toml
│   ├── data/
│   │   └── Streamlit_Input.csv <- 
│   └── images/
│       ├── Kalam-Vallay.jpg
│       ...                     <- Various images used in the app
│       └── stream.jpg
│
└── reports/
      └── WattsUp_Capstone_Presentation.pdf    <- Capstone presentation of Bootcamp
  ```
</details>

### Data

The data has to be collected from the [the IBM SkillsBuild Hydropower Climate Optimisation Challenge.](https://zindi.africa/competitions/ibm-skillsbuild-hydropower-climate-optimisation-challenge/data) 

- Data.zip - contains the hydropower generation data (energy output per source).

- Climate Data.zip - contains the daily weather and environmental variables.

- SampleSubmission.csv - template for final submission format.

### Notebooks 
- `01_data_preparation_and_exploration.ipynb` - demonstrates data cleaning, aggregation and exploration.  
- `02_naive_baseline_model.ipynb` - simple baseline for benchmarking.  
- `03_LightGBM_model_test.ipynb` - decision tree ensemble method based on building a strong learner by adding weak learners using gradient descent.  
- `04_Prophet_model_test.ipynb` - Metas Prophet model, time series model method using decomposition.  
- `06_ANN_model_test.ipynb` - deep learning artificial neural network with dense layers and monte carlo drop outs.
- `07_XGBoost_model_test.ipynb` - decision tree ensemble method based on sequential weak learners correcting errors of the previous one.


### Models (`src/models/`) 
- `ANN.py` – single python script based on 06_ANN_model_test.ipynb, includes parallelization for speed up.
- Models trained by `ANN.py` are saved as HDF5 filesin subfolders.

### Data (`src/data/`)
- Space for data files mentioned above, which will be used by `ANN.py`.

### Plots (`src/plots/`)
- Example plots are generated by `ANN.py` and saved in subfolders.

### Submissions (`src/submissions/`)
- Zindi submission files generated by `ANN.py` are saved in subfolders.

### Streamlit app
- Interactive dashboard for visualizing MLP model predictions vor individual households (users).

### Reports
- 'WattsUp_Capstone_Presentation.pdf' - Capstone project presentation for the Data Science bootcamp by neueFische.
---

## ⚙️ Setup

Follow these steps to set up and run the project locally.

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create a virtual environment

Recommended Python version 3.11.3

```bash
python -m venv venv
# Activate environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```



---

## 🚀 Running the Streamlit App

The Streamlit dashboard allows interactive exploration of model forecasts, historical hydropower data and values of the most important weather features.

Source of original data: Zindi (CC BY-SA 4.0). The dataset contained here is a transformed and aggregated derivative and does not include original raw data.

### Steps

1. Navigate to the `app/` directory and install its dependencies:

```bash
cd app
```
2. Launch the app:

```bash
streamlit run streamlit_app.py
```

This launches an interactive dashboard with multiple pages, powered by data in `app/data/`. This data contains a merged dataset that combines:

* Model predictions (from the trained MLP)
* Training data with historical hydropower and climate variables



---

## 📌 Notes

- `src/` contains the authoritative code for reproducibility.  
- `notebooks/` are explanatory and showcase data preparation, EDA, and baseline results.  
- Raw data should **not** be committed to GitHub.



---

## 🤝 Acknowledgements

- This work was done under the capstone project of the "Data Science, Machine Learning, and AI" bootcamp by [neuefische GmbH](https://www.neuefische.de).
- The data science challenge [IBM SkillsBuild Hydropower Climate Optimisation Challenge](https://zindi.africa/competitions/ibm-skillsbuild-hydropower-climate-optimisation-challenge) is provided by [Zindi Africa](https://zindi.africa) under CC-BY-SA 4.0 licence.



---

## ✨ Authors

- Noé Espinosa-Novo
- Gozal Jabrayilova 
- Patrick Kuntze
- Bernd Hermann
- Florencia Perachia

This project was developed collaboratively by the entire project team.  
**All members contributed equally to every stage of the project, including data preparation, modeling, visualization, app development, and reporting.**
