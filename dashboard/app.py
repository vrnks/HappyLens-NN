import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from keras.models import load_model
from openai import OpenAI
import matplotlib.pyplot as plt
import os   
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()  

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# === Load models ===
xgb_model = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
lstm_model = load_model(os.path.join(BASE_DIR, "lstm_model.keras"))
# === Load scaler and historical Ukraine data ===
scaler = joblib.load(os.path.join(BASE_DIR, "lstm_scaler.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "happiness_data.csv"))

# === Set feature names ===
features = ['Year', 'GDP', 'SocialSupport', 'LifeExpectancy', 'Freedom', 'Generosity', 'Corruption']

st.set_page_config(page_title="HappyLens-NN", layout="centered")
st.title("📊 HappyLens-NN - Ukraine 2024 Scenario Modeling")

st.markdown("Enter Ukraine’s socio-economic indicators for 2024:")

# === User Inputs ===
year = 2024
gdp = st.slider("GDP (0–2)", 0.0, 2.0, 0.8, step=0.01)
support = st.slider("Social Support (0–1)", 0.0, 1.0, 0.8, step=0.01)
life_expectancy = st.slider("Life Expectancy (0–1)", 0.0, 1.0, 0.7, step=0.01)
freedom = st.slider("Freedom (0–1)", 0.0, 1.0, 0.4, step=0.01)
generosity = st.slider("Generosity (0–1)", 0.0, 1.0, 0.2, step=0.01)
corruption = st.slider("Perception of Corruption (0–1)", 0.0, 1.0, 0.2, step=0.01)

# === Feature vector for 2024 ===
input_xgb = np.array([[gdp, support, life_expectancy, freedom, generosity, corruption]])

input_lstm = np.array([[year, gdp, support, life_expectancy, freedom, generosity, corruption]])

# === XGBoost Prediction ===
xgb_pred = xgb_model.predict(input_xgb)[0]

# === LSTM Prediction ===
# Take Ukraine data for 2022 and 2023
ukraine_prev = df[(df["Country"] == "Ukraine") & (df["Year"].isin([2022, 2023]))]
ukraine_prev = ukraine_prev.sort_values("Year")[features]

# Add current 2024 values
current_2024 = pd.DataFrame(input_lstm, columns=features)

# Create 3-year sequence
full_sequence = pd.concat([ukraine_prev, current_2024], ignore_index=True)
X_lstm = full_sequence.values.reshape(1, 3, len(features))

# Scale it
X_lstm_scaled = scaler.transform(X_lstm.reshape(-1, len(features))).reshape(1, 3, len(features))

# Predict
lstm_pred = lstm_model.predict(X_lstm_scaled, verbose=0)[0][0]

# === Display ===
st.markdown("### 🔎 Model Predictions for Ukraine (2024):")
st.success(f"**XGBoost Prediction:** {xgb_pred:.2f}")
st.info(f"**LSTM Prediction:** {lstm_pred:.2f}")


# === GenAI Explanation ===

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_explanation_openai(gdp, support, life_exp, freedom, generosity, corruption, xgb_pred, lstm_pred):
    prompt = f"""
The following values are user-defined inputs for a scenario simulation of Ukraine's happiness level in 2024. These values do not reflect real statistics but represent hypothetical socio-economic conditions selected by the user:

- GDP: {gdp:.2f}
- Social Support: {support:.2f}
- Life Expectancy: {life_exp:.2f}
- Freedom: {freedom:.2f}
- Generosity: {generosity:.2f}
- Perception of Corruption: {corruption:.2f}

Two AI models were used to predict Ukraine's Happiness Score for this scenario:
- XGBoost model prediction: {xgb_pred:.2f}
- LSTM model prediction: {lstm_pred:.2f}

Write a short explanation (4–6 sentences) analyzing how these simulated conditions might influence the predicted happiness level in Ukraine. Comment on why the two models might produce different results. Keep your explanation analytical and grounded in the input values.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a concise and insightful social analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

# === Streamlit button and display ===
st.markdown("### 💬 GenAI Explanation (OpenAI GPT-4)")
if st.button("🧠 Generate Explanation from GenAI"):
    with st.spinner("Generating explanation..."):
        explanation = generate_explanation_openai(
            gdp, support, life_expectancy, freedom, generosity, corruption, xgb_pred, lstm_pred
        )
        st.success("Done!")
        st.markdown(f"**🧠 Analytical Comment:**\n\n{explanation}")


# === Session scenario storage initialization ===
if 'scenarios' not in st.session_state:
    st.session_state.scenarios = []

# Button to add current scenario
if st.button("💾 Add Scenario for Comparison"):
    new_scenario = {
        'GDP': gdp,
        'Social Support': support,
        'Life Expectancy': life_expectancy,
        'Freedom': freedom,
        'Generosity': generosity,
        'Corruption': corruption,
        'XGBoost Prediction': xgb_pred,
        'LSTM Prediction': lstm_pred
    }
    
    if len(st.session_state.scenarios) >= 3:
        st.session_state.scenarios.pop(0)
    st.session_state.scenarios.append(new_scenario)
    st.success("Scenario added!")

indicators = ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
values = [gdp, support, life_expectancy, freedom, generosity, corruption]


actual_happiness_2024 = 4.68

if st.session_state.scenarios:
    st.markdown("### 📋 Saved Scenarios for Comparison")
    df_scenarios = pd.DataFrame(st.session_state.scenarios)
    st.dataframe(df_scenarios.style.format("{:.2f}"))

    fig, ax = plt.subplots(figsize=(7, 4))
    indices = list(range(1, len(df_scenarios) + 1))
    
    ax.plot(indices, df_scenarios['XGBoost Prediction'], 'o-', label="XGBoost Prediction", color='orange')
    ax.plot(indices, df_scenarios['LSTM Prediction'], 's-', label="LSTM Prediction", color='green')

    ax.axhline(y=actual_happiness_2024, color='red', linestyle='--', label='Actual 2024 Value')

    ax.set_xticks(indices)
    ax.set_xticklabels([f"Scenario {i}" for i in indices])
    ax.set_ylabel("Predicted Happiness Score")
    ax.set_title("Comparison of Predictions by Scenario")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)


# === Historical comparison ===
st.markdown("### 📈 Comparison between Real 2024 Values and User Inputs (Ukraine)")

real_2024 = df[(df["Country"] == "Ukraine") & (df["Year"] == 2024)][
    ['GDP', 'SocialSupport', 'LifeExpectancy', 'Freedom', 'Generosity', 'Corruption']
].iloc[0]

# Create a DataFrame to compare real 2024 values and user inputs
df_compare = pd.DataFrame({
    "Real 2024 Values": real_2024,
    "Input Values": pd.Series(values, index=indicators)
})

st.dataframe(df_compare.style.format("{:.2f}"))


def plot_feature_importance(model):
    fig, ax = plt.subplots(figsize=(8,5))
    xgb.plot_importance(model, ax=ax, max_num_features=10, importance_type='gain', show_values=False)
    plt.title("Feature Importance (Gain) - XGBoost")
    plt.tight_layout()
    return fig

st.markdown("### 🔑 Feature Importance (XGBoost)")

fig_imp = plot_feature_importance(xgb_model)
st.pyplot(fig_imp)

st.markdown("---")
st.caption("👁‍🗨 This tool uses socio-economic inputs to simulate the predicted happiness score for Ukraine in 2024 based on two AI models.")
