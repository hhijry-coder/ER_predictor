
# **Streamlit Web Application**

## **Overview**
This Streamlit web application allows users to upload datasets, train machine learning models (Traditional LSTM, Advanced LSTM, DNN), evaluate model performance with various metrics, and interactively visualize results. Additionally, the app provides an **interactive map** showing nearby hospitals and an **API integration** to fetch and display real-time weather information for any city or town.

---

## **Features**

1. **Data Upload**: 
   - Upload your datasets in CSV or XLSX format.
   - Select target and feature columns interactively.
   - Preprocess the data for training and scaling.

2. **Model Training**:
   - Choose from three models: **Traditional LSTM**, **Advanced LSTM**, and **DNN**.
   - Train your model on the uploaded dataset.
   - Model performance is displayed with metrics such as **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared (R²)**.

3. **Evaluation & Visualization**:
   - Visualize the **actual vs predicted** values and **residual plots** for model evaluation.
   - Get comprehensive performance metrics.
   - Perform error analysis and other visualizations interactively.

4. **Interactive Map**:
   - Enter a city name and find **nearby hospitals** using data from **OpenStreetMap**.
   - Customize the search radius in meters to adjust the distance for hospitals to display on the map.

5. **Weather API Integration**:
   - Enter any city name and fetch real-time **weather data** using the **OpenWeatherMap API**.
   - Display temperature, weather conditions, humidity, and wind speed for the selected city.

---

## **Usage Instructions**

### 1. **Local Installation**
To run this app locally on your machine, follow the steps below:

#### **Step 1: Clone the Repository**
```bash
git clone <repository_url>
cd <repository_directory>
```

#### **Step 2: Install Dependencies**
Ensure you have **Python 3.8+** installed and then install the required libraries.

```bash
pip install -r requirements.txt
```

#### **Step 3: Run the Streamlit Application**
```bash
streamlit run streamlit_app.py
```

This will launch the app in your default web browser. The Streamlit server will start and display the user interface.

---

### 2. **Deployment on Heroku**

#### **Step 1: Create a Heroku App**
If you don't have the Heroku CLI installed, install it first:
```bash
curl https://cli-assets.heroku.com/install.sh | sh
```
Then, create a Heroku app:
```bash
heroku login
heroku create your-app-name
```

#### **Step 2: Push the Code to Heroku**
```bash
git add .
git commit -m "Deploy Streamlit app to Heroku"
git push heroku master
```

#### **Step 3: Open the Application**
Once the deployment is complete, open the application with the following command:
```bash
heroku open
```

---

## **Configuration Files**

### 1. **`requirements.txt`**

This file contains all the Python packages required to run the application:

```txt
streamlit
numpy
pandas
tensorflow
scikit-learn
joblib
seaborn
matplotlib
plotly
folium
geopy
requests
streamlit-folium
```

### 2. **`Procfile`**

This file specifies the command to run your Streamlit app on Heroku:

```bash
web: streamlit run streamlit_app.py --server.port=$PORT
```

### 3. **`setup.sh`**

This shell script is used by Heroku to configure your Streamlit app properly:

```bash
mkdir -p ~/.streamlit/

echo "[server]
headless = true
port = \$PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

---

## **API Integrations**

### 1. **Weather API Integration**
This app uses the **OpenWeatherMap API** to fetch and display real-time weather information for any city or town.

- **API Key**: You need an API key from [OpenWeatherMap](https://openweathermap.org/api) to use the weather functionality.
- Replace the `API_KEY` variable in the `streamlit_app.py` with your API key:
  
  ```python
  API_KEY = 'your_openweathermap_api_key'
  ```

---

## **How to Use the Application**

1. **Navigate**: Use the **sidebar** to switch between different sections of the app:
   - **Data Upload**: Upload your dataset for model training.
   - **Model Training**: Choose and train the models on your data.
   - **Evaluation & Visualization**: See performance metrics and visualizations for your trained models.
   - **Interactive Map**: Enter a city name and find nearby hospitals.
   - **Weather API**: Enter a city name and get real-time weather data.

2. **Customize**: Adjust settings such as model type, feature columns, search radius, etc., interactively using the app controls.

---

## **Contributing**

If you'd like to contribute to the project:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Added feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

---

## **License**

This project is licensed under the MIT License.

---

## **Contact**

For questions, feedback, or suggestions, feel free to contact us at:

- **Email**: hmhijry@gmail.com
