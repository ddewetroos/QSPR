import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

st.title("Boiling Point Prediction: Final Graph and Statistics")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(df.head())

    # Linear regression model
    st.subheader("Linear Regression Model")
    X = df[["MW", "Wiener", "Z1", "Z2"]]
    X = sm.add_constant(X)
    y = df[["BP_K"]]
    model = sm.OLS(y, X).fit()
    st.write(model.summary())

    # Predicted vs observed plot
    pred_bp = model.fittedvalues.copy()
    fig, ax = plt.subplots()
    ax.scatter(pred_bp, df['BP_K'])
    ax.set_xlabel('Predicted Boiling Point (K)')
    ax.set_ylabel('Observed Boiling Point (K)')
    st.pyplot(fig)

    st.write("Regression statistics summary:")
    st.text(model.summary())
