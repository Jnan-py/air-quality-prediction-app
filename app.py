import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Air Quality Prediction", layout="wide", page_icon= ":clouds")
st.sidebar.title("Navigation")

with st.sidebar:
    page = option_menu(None, ["Home", "Data Visualization", "Model Training", "Prediction"], orientation= "vertical")

def load_data(file):
    try:
        df = pd.read_csv(file, sep=";", decimal=",", na_values=["NaN", "nan", "-200"])
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        df.columns = df.columns.str.strip()
        df.drop(columns=['Date', 'Time'], errors='ignore', inplace=True)
        df = df.apply(pd.to_numeric, errors='coerce')
        df.dropna(subset=['CO(GT)'], inplace=True)
        df.dropna(axis=1, how='all', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

df = load_data("Dataset\AirQualityUCI.csv")

if not df.empty:
    expected_columns = [col for col in df.columns if col not in ['CO(GT)']]
    
    if page == "Home":
        st.title("Air Quality Prediction App")
        st.write("### Preview of Dataset")
        st.dataframe(df.head())
        st.subheader("About the Dataset Columns")

        column_info = {
            "Column Name": [
                "Date",
                "Time",
                "CO(GT)",
                "PT08.S1(CO)",
                "NMHC(GT)",
                "C6H6(GT)",
                "PT08.S2(NMHC)",
                "NOx(GT)",
                "PT08.S3(NOx)",
                "NO2(GT)",
                "PT08.S4(NO2)",
                "PT08.S5(O3)",
                "T",
                "RH",
                "AH"
            ],
            "Data Type": [
                "Date",
                "Time",
                "Float",
                "Integer",
                "Float",
                "Float",
                "Integer",
                "Float",
                "Integer",
                "Float",
                "Integer",
                "Integer",
                "Float",
                "Float",
                "Float"
            ],
            "Description": [
                "Date of measurement",
                "Time of measurement",
                "Carbon Monoxide concentration (GT)",
                "PT08 sensor reading for CO",
                "Non-Methane Hydrocarbons concentration (GT)",
                "Benzene concentration (GT)",
                "PT08 sensor reading for NMHC",
                "Nitrogen Oxides concentration (GT)",
                "PT08 sensor reading for NOx",
                "Nitrogen Dioxide concentration (GT)",
                "PT08 sensor reading for NO2",
                "PT08 sensor reading for Ozone",
                "Temperature in degrees Celsius",
                "Relative Humidity",
                "Absolute Humidity"
            ]
        }

        df_info = pd.DataFrame(column_info)

        st.write(df_info)
    
    elif page == "Data Visualization":
        st.title("Data Visualization")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### CO(GT) Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df['CO(GT)'], bins=30, kde=True, ax=ax)
            plt.xlabel("CO Levels")
            plt.ylabel("Frequency")
            plt.title("Distribution of CO(GT)")
            st.pyplot(fig)
            
            st.write("### Box Plot")
            fig, ax = plt.subplots()
            sns.boxplot(data=df, orient="h", palette="coolwarm")
            st.pyplot(fig)
            
        with col2:
            st.write("### Correlation Matrix")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)  

            st.write("### Scatter Plot")
            fig, ax = plt.subplots()
            sns.scatterplot(x=df[expected_columns[0]], y=df['CO(GT)'], alpha=0.5)
            plt.xlabel(expected_columns[0])
            plt.ylabel("CO(GT)")
            plt.title("Scatter Plot")
            st.pyplot(fig)                      
            
            
        st.write("### Line Plot")
        fig, ax = plt.subplots()
        df.iloc[:100].plot(kind='line', ax=ax)
        plt.title("Line Plot of First 100 Entries")
        st.pyplot(fig)
        
    elif page == "Model Training":
        st.title("Model Training")
        X = df[expected_columns]
        y = df['CO(GT)']
        imputer = SimpleImputer(strategy="median")
        X_transformed = imputer.fit_transform(X)
        X = pd.DataFrame(X_transformed, columns=expected_columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_options = {
            "Linear Regression": LinearRegression(),
            "Lasso Regression": Lasso(),
            "Ridge Regression": Ridge(),
            "Random Forest": RandomForestRegressor()
        }
        
        selected_model_name = st.selectbox("Select a model", list(model_options.keys()))
        selected_model = model_options[selected_model_name]
        selected_model.fit(X_train, y_train)
        
        model_filename = f"{selected_model_name.replace(' ', '_')}.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(selected_model, file)
        st.write(f"Model saved as {model_filename}")
        
        y_pred = selected_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.write("### Model Performance")
        st.write(f"Mean Absolute Error: {mae:.4f}")
        st.write(f"Mean Squared Error: {mse:.4f}")
        st.write(f"R-Squared Score: {r2:.4f}")
    
    elif page == "Prediction":
        st.title("Make a Prediction")
        model_files = [f for f in os.listdir() if f.endswith('.pkl')]
        
        if model_files:
            selected_model_file = st.selectbox("Select a trained model", model_files)
            with open(selected_model_file, "rb") as file:
                selected_model = pickle.load(file)
            
            user_input = {}
            for col in expected_columns:
                user_input[col] = st.number_input(f"Enter value for {col}", value=float(df[col].median()))
            
            if st.button("Predict Air Quality"):
                input_df = pd.DataFrame([user_input])
                prediction = selected_model.predict(input_df)
                st.success(f"Predicted CO(GT) Level: {prediction[0]:.4f}")
        else:
            st.warning("No trained models found. Train a model first.")
