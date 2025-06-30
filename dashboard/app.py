import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from keras.models import load_model
from openai import OpenAI
import os
from dotenv import load_dotenv
import plotly.graph_objects as go

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

xgb_model = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
lstm_model = load_model(os.path.join(BASE_DIR, "lstm_model.keras"))
scaler = joblib.load(os.path.join(BASE_DIR, "lstm_scaler.pkl"))
df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "happiness_data.csv"))

features = ['Year', 'GDP', 'SocialSupport', 'LifeExpectancy', 'Freedom', 'Generosity', 'Corruption']

st.set_page_config(page_title="HappyLens-NN", layout="wide")
st.title("📊 HappyLens-NN - Ukraine 2024 Scenario Modeling")
st.markdown("Enter Ukraine’s socio-economic indicators for 2024:")

col_left, col_right = st.columns([1, 1.5])

with col_left:
    year = 2024
    gdp = st.slider("GDP (0–2)", 0.0, 2.0, 0.8, step=0.01)
    support = st.slider("Social Support (0–1)", 0.0, 1.0, 0.8, step=0.01)
    life_expectancy = st.slider("Life Expectancy (0–1)", 0.0, 1.0, 0.7, step=0.01)
    freedom = st.slider("Freedom (0–1)", 0.0, 1.0, 0.4, step=0.01)
    generosity = st.slider("Generosity (0–1)", 0.0, 1.0, 0.2, step=0.01)
    corruption = st.slider("Perception of Corruption (0–1)", 0.0, 1.0, 0.2, step=0.01)

    input_xgb = np.array([[gdp, support, life_expectancy, freedom, generosity, corruption]])
    input_lstm = np.array([[year, gdp, support, life_expectancy, freedom, generosity, corruption]])

    xgb_pred = xgb_model.predict(input_xgb)[0]

    ukraine_prev = df[(df["Country"] == "Ukraine") & (df["Year"].isin([2022, 2023]))]
    ukraine_prev = ukraine_prev.sort_values("Year")[features]
    current_2024 = pd.DataFrame(input_lstm, columns=features)
    full_sequence = pd.concat([ukraine_prev, current_2024], ignore_index=True)
    X_lstm = full_sequence.values.reshape(1, 3, len(features))
    X_lstm_scaled = scaler.transform(X_lstm.reshape(-1, len(features))).reshape(1, 3, len(features))
    lstm_pred = lstm_model.predict(X_lstm_scaled, verbose=0)[0][0]

    st.markdown("### 🔎 Model Predictions for Ukraine (2024):")
    st.success(f"**XGBoost Prediction:** {xgb_pred:.2f}")
    st.info(f"**LSTM Prediction:** {lstm_pred:.2f}")

with col_right:
    if 'scenarios' not in st.session_state:
        st.session_state.scenarios = []

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

    if st.session_state.scenarios:
        st.markdown("### 📋 Saved Scenarios for Comparison")
        df_scenarios = pd.DataFrame(st.session_state.scenarios)
        styled_table = df_scenarios.style.format("{:.2f}") \
            .set_table_styles([
                {'selector': 'thead th', 'props': [('background-color', '#2c3e50'), ('color', 'white'), ('font-weight', 'bold')]},
                {'selector': 'tbody tr:hover', 'props': [('background-color', '#ecf0f1')]},
                {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f7f9f9')]},
                {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
            ])
        st.dataframe(styled_table)

        def plot_comparison_plotly(df_scenarios):
            indices = list(range(1, len(df_scenarios) + 1))
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=indices,
                y=df_scenarios['XGBoost Prediction'],
                mode='lines+markers',
                name='XGBoost Prediction',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=10)
            ))

            fig.add_trace(go.Scatter(
                x=indices,
                y=df_scenarios['LSTM Prediction'],
                mode='lines+markers',
                name='LSTM Prediction',
                line=dict(color='#3A5FCD', width=3),
                marker=dict(size=10, symbol='square')
            ))

            fig.add_shape(
                type='line',
                x0=1, x1=len(df_scenarios),
                y0=4.68, y1=4.68,
                line=dict(color='#a93226', width=2, dash='dash'),
                xref='x',
                yref='y',
            )

            fig.add_trace(go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color='#a93226', width=2, dash='dash'),
                name='Actual 2024 Value'
            ))

            fig.update_layout(
                title="Comparison of Predictions by Scenario",
                xaxis=dict(
                    tickmode='array',
                    tickvals=indices,
                    ticktext=[f"Scenario {i}" for i in indices],
                    showgrid=True,
                    gridcolor='LightGray'
                ),
                yaxis=dict(title="Predicted Happiness Score", showgrid=True, gridcolor='LightGray'),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(
                    bgcolor='rgba(255,255,255,0)',
                    borderwidth=0,
                    font=dict(size=12),
                    yanchor="bottom",
                    y=-0.3,
                    xanchor="center",
                    x=0.5,
                    orientation="h",
                    traceorder="normal"
                ),
                font=dict(size=14),
                margin=dict(l=40, r=40, t=60, b=80)
            )
            return fig


        fig_plotly = plot_comparison_plotly(df_scenarios)
        st.plotly_chart(fig_plotly, use_container_width=True)

st.markdown("---")
st.markdown("### 💬 GenAI Explanation (OpenAI GPT-4)")

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


if st.button("🧠 Generate Explanation from GenAI"):
    with st.spinner("Generating explanation..."):
        explanation = generate_explanation_openai(
            gdp, support, life_expectancy, freedom, generosity, corruption, xgb_pred, lstm_pred
        )
        st.success("Done!")
        st.markdown(f"**🧠 Analytical Comment:**\n\n{explanation}")

st.markdown("---")
st.markdown("### 📈 Comparison between Real 2024 Values and User Inputs (Ukraine)")

real_2024 = df[(df["Country"] == "Ukraine") & (df["Year"] == 2024)][
    ['GDP', 'SocialSupport', 'LifeExpectancy', 'Freedom', 'Generosity', 'Corruption']
].iloc[0]

indicators = ['GDP', 'Social Support', 'Life Expectancy', 'Freedom', 'Generosity', 'Corruption']
values = [gdp, support, life_expectancy, freedom, generosity, corruption]

df_compare = pd.DataFrame({
    "Real 2024 Values": real_2024,
    "Input Values": pd.Series(values, index=indicators)
})

styled_compare = df_compare.style.format("{:.2f}") \
    .set_table_styles([
        {'selector': 'thead th', 'props': [('background-color', '#34495e'), ('color', 'white'), ('font-weight', 'bold')]},
        {'selector': 'tbody tr:hover', 'props': [('background-color', '#ecf0f1')]},
        {'selector': 'tbody tr:nth-child(even)', 'props': [('background-color', '#f7f9f9')]},
        {'selector': 'tbody tr:nth-child(odd)', 'props': [('background-color', 'white')]},
    ])
st.dataframe(styled_compare)

def plot_feature_importance_plotly(model):
    importance = model.get_booster().get_score(importance_type='gain')
    items = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    features, scores = zip(*items)

    fig = go.Figure(go.Bar(
        x=scores[::-1],
        y=features[::-1],
        orientation='h',
        marker=dict(color='#2ecc71'),  
    ))

    fig.update_layout(
        title="Feature Importance (Gain) - XGBoost",
        xaxis_title="Gain",
        yaxis_title="Feature",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14),
        margin=dict(l=100, r=40, t=50, b=40)
    )
    return fig

st.markdown("### 🔑 Feature Importance (XGBoost)")
fig_imp = plot_feature_importance_plotly(xgb_model)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown("---")
st.caption("👁‍🗨 This tool uses socio-economic inputs to simulate the predicted happiness score for Ukraine in 2024 based on two AI models.")
