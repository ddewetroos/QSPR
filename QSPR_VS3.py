import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from mordred import Calculator, descriptors, WienerIndex, ZagrebIndex
import statsmodels.api as sm

# Title
st.title("Chemical Descriptor Analysis and Boiling Point Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.write(df.head())

    # Descriptor calculation setup
    calc = Calculator(descriptors, ignore_3D=True)
    wiener_index = WienerIndex.WienerIndex()
    zagreb_index1 = ZagrebIndex.ZagrebIndex(version=1)
    zagreb_index2 = ZagrebIndex.ZagrebIndex(version=2)

    # Calculate descriptors
    result_Wiener = []
    result_Z1 = []
    result_Z2 = []

    for index, row in df.iterrows():
        SMILE = row['smiles']
        mol = Chem.MolFromSmiles(SMILE)
        result_Wiener.append(wiener_index(mol))
        result_Z1.append(zagreb_index1(mol))
        result_Z2.append(zagreb_index2(mol))

    df['Wiener'] = result_Wiener
    df['Z1'] = result_Z1
    df['Z2'] = result_Z2

    st.write("Data with calculated descriptors:")
    st.write(df.head())

    # Scatter plots
    st.subheader("Scatter Plots")
    fig1, ax1 = plt.subplots()
    ax1.scatter(df['Wiener'], df['BP_K'])
    ax1.set_xlabel('Wiener Index')
    ax1.set_ylabel('Boiling Point (K)')
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(df['Z1'], df['BP_K'])
    ax2.set_xlabel('Zagreb Index 1')
    ax2.set_ylabel('Boiling Point (K)')
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    ax3.scatter(df['Z2'], df['BP_K'])
    ax3.set_xlabel('Zagreb Index 2')
    ax3.set_ylabel('Boiling Point (K)')
    st.pyplot(fig3)

    # Save updated data
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download updated data", csv, "updated_data.csv", "text/csv")

    # Linear regression model
    st.subheader("Linear Regression Model")
    X = df[["MW", "Wiener", "Z1", "Z2"]]
    X = sm.add_constant(X)
    y = df[["BP_K"]]
    model = sm.OLS(y, X).fit()
    st.write(model.summary())

    # Predicted vs observed plot
    pred_bp = model.fittedvalues.copy()
    fig4, ax4 = plt.subplots()
    ax4.scatter(pred_bp, df['BP_K'])
    ax4.set_xlabel('Predicted Boiling Point (K)')
    ax4.set_ylabel('Observed Boiling Point (K)')
    st.pyplot(fig4)

    st.write("Regression statistics summary:")
    st.text(model.summary())

    # Alternative model without Z2
    st.subheader("Alternative Model without Z2")
    X_alt = df[["MW", "Wiener", "Z1"]]
    X_alt = sm.add_constant(X_alt)
    model_alt = sm.OLS(y, X_alt).fit()
    st.write(model_alt.summary())
