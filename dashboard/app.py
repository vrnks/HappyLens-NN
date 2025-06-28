import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from keras.models import load_model
from openai import OpenAI
import matplotlib.pyplot as plt
import os   
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
st.title("📊 HappyLens-NN — Ukraine 2024 Scenario Modeling")

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


indicators = ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
values = [gdp, support, life_expectancy, freedom, generosity, corruption]

# === Happiness trend plot ===
st.markdown("### 📉 Happiness Score Trend for Ukraine (2020–2024)")

# Get actual values (2020–2024)
happiness_trend = df[(df["Country"] == "Ukraine") & (df["Year"].between(2020, 2024))][["Year", "HappinessScore"]]
happiness_trend = happiness_trend.sort_values("Year")

# Extract real 2024 value
real_2024 = happiness_trend[happiness_trend["Year"] == 2024]["HappinessScore"].values[0]

# Build plot
fig, ax = plt.subplots(figsize=(7, 4))

# Line: actual values
ax.plot(happiness_trend["Year"], happiness_trend["HappinessScore"], marker='o', label="📌 Real Score", color="black")

# XGBoost prediction
ax.plot(2024, xgb_pred, marker='D', markersize=7, label="🤖 XGBoost Prediction", color="orange")

# LSTM prediction
ax.plot(2024, lstm_pred, marker='s', markersize=7, label="🧠 LSTM Prediction", color="green")

# Decorations
ax.set_xticks([2020, 2021, 2022, 2023, 2024])
ax.set_ylabel("Happiness Score")
ax.set_title("Happiness Score in Ukraine (2020–2024)")
ax.grid(True)
ax.legend()

st.pyplot(fig)

# === Historical comparison ===
st.markdown("### 📈 Comparison with Previous Years (Ukraine)")

hist_years = [2022, 2023]
hist_df = df[(df["Country"] == "Ukraine") & (df["Year"].isin(hist_years))]
hist_means = hist_df[['GDP', 'SocialSupport', 'LifeExpectancy', 'Freedom', 'Generosity', 'Corruption']].mean()

df_compare = pd.DataFrame({
    "2022–2023 Avg": hist_means,
    "2024 Input": pd.Series(values, index=indicators)
})

st.dataframe(df_compare.style.format("{:.2f}"))

st.markdown("---")
st.caption("👁‍🗨 This tool uses socio-economic inputs to simulate the predicted happiness score for Ukraine in 2024 based on two AI models.")
