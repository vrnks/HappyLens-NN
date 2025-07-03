
# HappyLens NN

## Project Overview

**HappyLens NN** is a data science project exploring happiness levels across countries through statistical analysis, machine learning, and predictive scenario modeling. Using real data from the World Happiness Report, it combines exploratory analysis, model building (Random Forest, Gradient Boosting, XGBoost, Neural Network, LSTM) to forecast happiness scores based on socio-economic indicators, and scenario-based simulations. The focus is on modeling Ukraine’s happiness score for 2024 - and exploring how changes in key factors like GDP, Social Support, and Freedom would affect the outcome.


## Objectives

- Understand global happiness trends using historical data.
- Explore and visualize factors influencing happiness across the world.
- Build regression models (including simple neural nets) to predict national happiness scores.
- Simulate “what-if” scenarios to explore how changes in factors influence outcomes.
- Identify the most impactful drivers of national well-being.
- Evaluate GenAI's utility for interpreting predictive scenarios.


## Dataset

- **Rows:** 1,956  
- **Columns:** 10 (including year, country, happiness score, and 6 predictors)  
- **Countries:** ~140  
- **Years covered:** 2011-2024  

## Project Structure

```
HappyLens_NN/
├── data/                           # Dataset and preprocessing notebook
│ ├── World Happiness Report.csv
│ ├── happiness_data.csv            # Cleaned and processed dataset
│ └── data_preprocessing.ipynb      # Notebook for data cleaning and transformation
├── data_analysis/                  
│ └── data_analysis.ipynb           # EDA, visualizations
├── model_what_if/                  # Predictive models and what-if simulations
│ ├── happiness_predictor.ipynb     # Builds and evaluates ML models (linear, RF, XGBoost)
│ ├── happiness_predictor_nn.ipynb  # LSTM model and what-if simulations
│ ├── model_choosing.ipynb          # Experiments with NN architectures and selects the best one
│ ├── model_comparing.ipynb         # Comparing XGBoost and LSTM
│ ├── xgb_scenarios.json            # What-if scenario results using XGBoost model
│ └── lstm_scenarios.json           # What-if scenario results using LSTM model
├── dashboard/                      # Streamlit dashboard code
│   └── app.py
├── README.md
├── requirements.txt                # Python dependencies
```


## Features

- **Exploratory Data Analysis (EDA)**: Correlation matrix, visual summaries.
- **Prediction Models**:
  - Traditional regressors:
    - Linear Regression,
    - Random Forest,
    - Gradient Boosting,
    - XGBoost.
  - Neural approaches:
    - Feedforward Neural Networks (Keras) - multiple variants with dropout, feature engineering, and regularization.
    - LSTM (Recurrent Neural Network) - models temporal sequences of happiness data across years.
- **Scenario Simulation**: Predict how score changes with hypothetical improvements.
- **Feature Engineering**: Removal of irrelevant features. Standardization via StandardScaler.


## Tech Stack

- **Language**: Python 3.11
- **Jupyter Notebook** for analysis and modeling
- **Interface**: Streamlit
- **Data**: World Happiness Report 2024
- **ML Libraries**: `scikit-learn`, `xgboost`, `keras`, `tensorflow`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Data Handling**: `pandas`, `numpy`, `json`
- **OpenAI** (for explanation generation via GenAI): openai API (GPT-4)


## What GenAI Does Here:

**Scenario Explanation Generation**: after forecasting how happiness would change under specific variable adjustments (e.g., +10% GDP), GenAI (via OpenAI models) generates human-readable, contextual explanations for:
- Why the score changed the way it did
- Which variables had the strongest impact
- What trade-offs or hidden effects might emerge
  
![5B530865-55C1-4B23-A9F7-08AAB2ED1E3C](https://github.com/user-attachments/assets/00d6cc93-4811-436d-87a0-2f51fafe2a42)

## Results

- Ukraine's 2024 happiness score was predicted.
- Feature importance analysis across models revealed that:
Social Support, Freedom to Make Life Choices, and GDP per capita were consistently top predictors. However, freedom and support showed stronger marginal effects on predicted happiness in what-if simulations than raw GDP growth.
- The best-performing model overall was **XGBoost** for standard regression, and **LSTM** for time-aware prediction.
- Enabled simulations of “what-if” scenarios (e.g., increased GDP, improved social support, freedom).

## Models Compared

| Indicator Change         | XGBoost Prediction | LSTM Prediction |
|--------------------------|--------------------|-----------------|
| GDP ↑ by 10%             | 5.351              | 5.706           |
| Social Support ↑ by 0.1  | 5.461              | 5.755           |
| Freedom ↑ by 0.1         | 5.612              | 5.775           |

These results suggest that increasing perceived freedom would have a greater positive impact on happiness than an equivalent increase in economic output. This insight challenges conventional economic-first approaches and emphasizes the value of civil liberties and autonomy in national well-being.

The best-performing model for static regression tasks was XGBoost, while LSTM excelled in capturing temporal shifts and trends in national happiness over time.
Using Generative AI, each scenario was accompanied by an automatically generated natural language explanation that contextualized the predicted change, identified key influencing variables, and suggested plausible socio-political interpretations. 
              
                         
## How to Use

### Streamlit Dashboard (Recommended)

You can explore the dashboard either by:

* Opening the hosted app via this link:		[HappyLens NN](https://happylens-nn.streamlit.app)

* Or running it locally from code:

```bash
cd HappyLens-NN/dashboard
streamlit run app.py
```

1. Adjust socio-economic indicators for Ukraine in 2024.
2. View predictions from both XGBoost and LSTM models.
3. Generate GenAI-powered explanations for the predicted outcomes.
4. Compare multiple scenarios visually and numerically.
5. See how your simulated inputs differ from actual 2024 data.
6. Explore XGBoost feature importance interactively.

### Install Requirements

```bash
pip install -r requirements.txt
```

### Explore EDA and Modeling

Open the notebooks in Jupyter:

- `data_analysis/data_analysis.ipynb`: Explore global patterns.
- `model_what_if/happiness_predictor.ipynb`: Trains and evaluates baseline ML models - Linear Regression, Random Forest, XGBoost - and analyzes feature importance.
- `model_what_if/happiness_predictor_nn.ipynb`: Builds and evaluates the LSTM model and runs what-if scenario simulations.
- `model_what_if/model_comparing.ipynb`: Compares XGBoost and LSTM models to assess time-aware vs. standard predictions.
- `model_what_if/model_choosing.ipynb`: Builds and tunes neural networks and LSTM models for time-aware prediction. Identifies the best-performing model for scenario analysis.

## Inspiration

This project aims to go beyond generic happiness rankings by making the findings interpretable and relevant to policy and individual-level planning.
	***"There is no universal formula for happiness - but there can be a data-driven guide to help find your own."***


