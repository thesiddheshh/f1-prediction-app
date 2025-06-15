# DRS Diva: F1 Race Predictor 🏁

A data-driven Formula 1 race outcome predictor using FastF1, machine learning models, weather integration, and real-world racing insights.

This project predicts podium finishes and driver points based on clean air pace, qualifying performance, weather, and other contextual race features. It includes a fully functional [Streamlit](https://streamlit.io/) dashboard for interactive exploration and simulation.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
- [Sample Predictions](#sample-predictions)
- [Future Improvements](#future-improvements)
- [FAQ](#faq)
- [License](#license)
- [Contact](#contact)

---

## 🔍 Overview

Formula 1 is a complex sport where race outcomes are determined by an intricate blend of driver skill, car performance, team strategy, and environmental conditions. Traditional statistics or fan predictions often miss these multidimensional interactions.

This project takes a data science approach to forecasting race results by:
- Leveraging **FastF1 telemetry and session data**
- Incorporating **weather conditions via external API**
- Using **Gradient Boosting Regression models** for point and podium predictions
- Offering **interactive insights via Streamlit UI**

It's built for:
- Data science students exploring ML in sports
- F1 fans wanting to simulate race outcomes
- Developers interested in telemetry and live sports data integration

---

## ✨ Features

- **Race Weekend Data**: Pulls telemetry, lap times, and qualifying data using FastF1.
- **Weather API Integration**: Incorporates real-time or historical weather into prediction models.
- **Missing Data Handling**: Uses imputation strategies for partial driver telemetry.
- **ML Modeling**: Uses Scikit-learn's GradientBoostingRegressor for point and position predictions.
- **Clean Air Pace Analysis**: Focuses on driver potential without traffic interference.
- **Podium Predictor**: Simulates and visualizes expected podium finishers.
- **Dashboard Interface**: Upload data, view model performance, and get predictions in a user-friendly web app.

---

## ⚙️ Tech Stack

| Component         | Technology                     |
|------------------|---------------------------------|
| Data Source       | [FastF1](https://github.com/theOehrly/Fast-F1) |
| Web App           | Streamlit                      |
| ML Modeling       | Scikit-learn                   |
| Visualization     | Matplotlib, Streamlit          |
| Weather Data      | OpenWeather API (optional)     |
| Packaging         | Python                         |
| Deployment        | Streamlit Cloud                |

---

## 🗂️ Project Structure

```
f1-race-predictor/
│
├── app.py                         # Main Streamlit app
├── requirements.txt               # List of Python dependencies
├── README.md                      # Project documentation
│
├── src/                           # Modular Python scripts
│   ├── data_loader.py             # Load and cache race data from FastF1
│   ├── preprocess.py              # Clean and engineer features
│   ├── model.py                   # Train and evaluate ML models
│   ├── weather.py                 # API calls and parsing for weather info
│   └── visualization.py           # Plotting functions for telemetry and results
│
├── experiments/                   # Experimental notebooks or scripts
│   ├── modeling_arima.py          # Placeholder: time series models (optional)
│   └── compare_models.py          # Baseline comparisons, testing alternatives
│
├── f1_cache/                      # FastF1 cache directory (local use only)
│
├── outputs/                       # Plots, logs, and model outputs
│   ├── predictions/               # CSVs or charts with predicted positions
│   └── plots/                     # MAE charts, qualifying gaps, etc.
│
└── assets/                        # Icons, logos, static images (optional)
```

---

## 🖥️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/f1-race-predictor.git
cd f1-race-predictor
```

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate          # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Enable FastF1 Caching

```python
import fastf1
fastf1.Cache.enable_cache("f1_cache")
```

This ensures race telemetry is cached locally to avoid repeated downloads.

---

## ▶️ Running the App

To launch the app on your machine:

```bash
streamlit run app.py
```

The app will open in your browser (usually at http://localhost:8501).

---

## ☁️ Deployment on Streamlit Cloud

To deploy this app publicly:

### Step 1: Push to GitHub

Ensure your code and `requirements.txt` are committed to GitHub.

### Step 2: Go to Streamlit Cloud

Visit: https://streamlit.io/cloud

### Step 3: Create a New App

- Choose your GitHub repo and branch
- Set the entry point to `app.py`

### Step 4: Add Secrets (Optional for Weather)

Go to “Secrets” tab and add:

```toml
WEATHER_API_KEY = "your_openweather_api_key"
```

### Step 5: Deploy

Click "Deploy" and Streamlit will handle the rest.

Your app will be live at:

```
https://<your-username>-<repo-name>.streamlit.app
```

---

## 📈 Sample Predictions

Once the app is live or running locally, you can:

- Select a 2024 race weekend (e.g., Monaco GP)
- Upload qualifying performance if needed
- View model-predicted finishing positions
- Compare predictions with real-world results
- Analyze clean air pace vs actual outcome

---

## 🚧 Future Improvements

- Integrate LSTM/RNN models for better race-length prediction
- Include pit strategy, tyre degradation models
- Scrape and incorporate sector-specific pace data
- Add real-time updates during live races
- Allow user-defined race simulations (e.g., weather, crash probabilities)

---

## ❓ FAQ

**Q: Can I use this for real-time betting or live predictions?**  
A: No. This project is for educational and exploratory purposes only. It is not optimized or legally intended for gambling or betting use.

**Q: Why are predictions sometimes inaccurate?**  
A: Race outcomes involve countless unpredictable variables (safety cars, rain, DNFs). The model uses only historical and available telemetry + weather data.

**Q: Does it work with Sprint weekends?**  
A: Currently, the app is optimized for standard race weekends. Sprint sessions may require manual handling.

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for details.

---

## 📬 Contact

**Developer:** Siddh Wagawad  
**GitHub:** [@thesiddheshh](https://github.com/thesiddheshh)  
**Instagram:** [@thesiddheshh](https://instagram.com/thesiddheshh)  
**Email:** *[siddhwagawad@gmail.com]*

---

> If you find this project useful, feel free to ⭐ star it on GitHub or share it with fellow F1 enthusiasts!
