# DRS Diva: F1 Race Predictor

A data-driven Formula 1 outcome predictor using FastF1 telemetry, machine learning models, weather integration, and contextual racing features.

This project forecasts podium finishes and driver points using clean air pace, qualifying performance, weather conditions, and more. It includes an interactive [Streamlit](https://streamlit.io/) dashboard for exploration and simulation.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [2025 Race Predictions vs Actuals](#-📊-2025-race-predictions-vs-actuals)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Deployment on Streamlit Cloud](#deployment-on-streamlit-cloud)
- [Sample Predictions](#sample-predictions)
- [Future Improvements](#future-improvements)
- [FAQ](#faq)
- [License](#license)
- [Contact](#contact)

---

## Overview

Formula 1 is a multifaceted sport influenced by driver skill, car performance, team strategy, and environmental conditions. This project adopts a data science approach to race outcome forecasting by:

- Using **FastF1 telemetry and session data**
- Integrating **weather features** via API
- Training **machine learning models** for point and podium prediction
- Providing an **interactive Streamlit dashboard**

Ideal for:

- Data science students exploring ML in sports
- F1 fans simulating race outcomes
- Developers working with telemetry and real-time sports data

---

## Features

- **Telemetry Extraction**: Accesses lap times, sector data, and qualifying metrics via FastF1.
- **Weather Integration**: Adds real-time or historical weather data using the OpenWeather API.
- **Data Imputation**: Handles incomplete telemetry using intelligent fill strategies.
- **Machine Learning Models**: Uses Scikit-learn’s `GradientBoostingRegressor` for driver performance predictions.
- **Clean Air Pace**: Estimates true driver speed potential excluding traffic effects.
- **Podium Simulation**: Predicts likely podium finishers based on model results.
- **Streamlit Dashboard**: Web app interface for data upload, predictions, and visualization.

---

## Tech Stack

| Component        | Technology                                  |
|------------------|---------------------------------------------|
| Data Source      | [FastF1](https://github.com/theOehrly/Fast-F1) |
| ML Modeling      | Scikit-learn                                |
| Web App          | Streamlit                                   |
| Weather API      | OpenWeather API                             |
| Visualization    | Matplotlib, Streamlit plotting modules      |
| Deployment       | Streamlit Cloud                             |
| Language         | Python                                      |

---

## Project Structure

```
f1-race-predictor/
│
├── app.py                        # Main Streamlit application
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation
│
├── src/                          # Core logic and modular scripts
│   ├── data_loader.py            # FastF1 session/telemetry loading
│   ├── preprocess.py             # Feature engineering, cleaning
│   ├── model.py                  # Training, prediction logic
│   ├── weather.py                # Weather API integration
│   └── visualization.py          # Telemetry and result visualizations
│
├── experiments/                  # Model testing and exploratory scripts
│   ├── modeling_arima.py         # (Optional) time series models
│   └── compare_models.py         # Alternative models & evaluation
│
├── f1_cache/                     # FastF1 telemetry cache
├── outputs/                      # Model outputs and plots
│   ├── predictions/              # CSVs, JSONs of race predictions
│   └── plots/                    # Performance graphs and race analytics
│
└── assets/                       # Icons, logos, or branding (optional)
```

---

### 📊 2025 Race Predictions vs Actuals

Race Weekend       | Predicted Top 3      | Actual Top 3         | Top‑3 Accuracy
-------------------|----------------------|-----------------------|----------------
Bahrain GP         | VER, LEC, HAM        | PIA, RUS, NOR         | ✅ 1/3 (33%)
Saudi GP           | VER, LEC, PIA        | PIA, VER, LEC         | ✅ 3/3 (100%)
Australian GP      | LEC, RUS, NOR        | NOR, VER, RUS         | ✅ 2/3 (66%)
Japanese GP        | VER, LEC, NOR        | VER, NOR, PIA         | ✅ 2/3 (66%)
Chinese GP         | PIA, NOR, RUS        | PIA, NOR, RUS         | ✅ 3/3 (100%)
Miami GP           | VER, PIA, NOR        | PIA, NOR, VER         | ✅ 3/3 (100%)
Emilia-Romagna GP  | VER, LEC, PIA        | VER, NOR, PIA         | ✅ 2/3 (66%)
Monaco GP          | LEC, NOR, PIA        | LEC, NOR, PIA         | ✅ 3/3 (100%)
Canadian GP        | HAM, VER, RUS        | RUS, VER, ANTONELLI   | ✅ 2/3 (66%)
Spanish GP         | VER, HAM, LEC        | PIA, NOR, LEC         | ✅ 1/3 (33%)
Austrian GP        |                      | —                     | —
British GP         | HAM, VER, NOR        | —                     | —
Hungarian GP       | PIA, NOR, HAM        | —                     | —
Belgian GP         | RUS, PIA, HAM        | —                     | —
Dutch GP           | LEC, VER, PIA        | —                     | —
Italian GP         | LEC, NOR, PIA        | —                     | —
Azerbaijan GP      | PIA, LEC, VER        | —                     | —
Singapore GP       | VER, PIA, LEC        | —                     | —
US GP (Austin)     |                      | —                     | —
Mexico GP          | LEC, PIA, RUS        | —                     | —
Brazilian GP       |                      | —                     | —
Las Vegas GP       | RUS, HAM, LEC        | —                     | —
Qatar GP           |                      | —                     | —
Abu Dhabi GP       | LEC, SAI, RUS        | —                     | —

---

---

### 🚀 Try DRS Diva Now

Live race predictor and simulation dashboard:  
🔗 [**drs-diva.streamlit.app**](https://drs-diva.streamlit.app)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/f1-race-predictor.git
cd f1-race-predictor
```

### 2. (Optional) Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate         # For macOS/Linux
venv\Scripts\activate            # For Windows
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

---

## Running the App

To launch the app locally:

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser to view the dashboard.

---

## Deployment on Streamlit Cloud

### Step 1: Push to GitHub

Ensure `app.py`, `requirements.txt`, and necessary folders are pushed to your GitHub repository.

### Step 2: Deploy via Streamlit Cloud

- Go to [Streamlit Cloud](https://streamlit.io/cloud)
- Connect your GitHub repository
- Set `app.py` as the entry point

### Step 3: (Optional) Add Weather API Key

Go to the **"Secrets"** section and add:

```toml
WEATHER_API_KEY = "your_openweather_api_key"
```

### Step 4: Deploy

Click **Deploy**. Your app will be live at:

```
https://<your-username>-<repo-name>.streamlit.app
```

---

## Sample Predictions

- Choose a recent race weekend (e.g., Monaco 2025)
- Upload or simulate qualifying/telemetry data
- View predicted podium finishers and driver points
- Compare predictions with actual outcomes
- Analyze clean air performance and visual telemetry

---

## Future Improvements

- Incorporate RNN/LSTM models for dynamic race length forecasting
- Model tyre degradation, pit strategy, and track evolution
- Add real-time race telemetry updates
- Integrate sector-specific pace breakdowns
- Allow users to simulate custom race scenarios (e.g., rain, safety car events)

---

## FAQ

**Q: Can this be used for live betting or gambling?**  
A: No. This tool is for academic, educational, and fan engagement purposes only.

**Q: Why aren't predictions always accurate?**  
A: F1 races are unpredictable. Models do not currently account for incidents like crashes, pit errors, or DNFs.

**Q: Is it optimized for sprint weekends?**  
A: Currently, it supports standard race weekends. Sprint formats may require manual handling.

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contact

**Developer:** Siddh Wagawad  
**GitHub:** [https://github.com/thesiddheshh](https://github.com/thesiddheshh)  
**Email:** siddhwagawad@gmail.com  
**Instagram:** [@thesiddheshh](https://instagram.com/thesiddheshh)

---

> If you found this project useful, feel free to star ⭐ the repository and share it with F1 fans and ML enthusiasts.
