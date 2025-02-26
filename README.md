# Air Quality Prediction App

This is a Streamlit-based web application designed to predict air quality (specifically the CO(GT) level) using machine learning models. The application allows users to explore the dataset, visualize key insights, train various regression models, and make predictions based on custom input values.

## Features

- **Home:** Preview the air quality dataset and view detailed information about its columns.
- **Data Visualization:** Interactive plots including histograms, box plots, heatmaps, scatter plots, and line plots to explore the data.
- **Model Training:** Train regression models (Linear Regression, Lasso, Ridge, Random Forest) on the dataset and evaluate model performance.
- **Prediction:** Make predictions using the trained models by entering custom feature values.
- **Navigation:** Simple page navigation using an option menu in the sidebar.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/Jnan-py/air-quality-prediction-app.git
   cd air-quality-prediction-app
   ```

````

2. **Create a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare the Dataset:**

   - Ensure the dataset `AirQualityUCI.csv` is placed in the `Dataset` folder.
   - The app expects the dataset to be in the following path: `Dataset/AirQualityUCI.csv`

2. **Run the Application:**

   ```bash
   streamlit run app.py
   ```

3. **Navigation:**
   - Use the sidebar to navigate between the Home, Data Visualization, Model Training, and Prediction pages.
   - Explore the dataset, visualize the data, train a model of your choice, and make predictions based on your input.

## Project Structure

```
air-quality-prediction-app/
│
├── app.py                   # Main Streamlit application
├── Dataset/
│   └── AirQualityUCI.csv    # Dataset file
├── model_names.pkl         # Trained model files will be saved here (e.g., Linear_Regression.pkl)
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

## Technologies Used

- **Python** for data processing and model training.
- **Streamlit** for the web interface.
- **Pandas & NumPy** for data manipulation.
- **Matplotlib & Seaborn** for data visualization.
- **Scikit-Learn** for machine learning (model training, evaluation, and imputation).
- **streamlit-option-menu** for sidebar navigation.
````
