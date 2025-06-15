import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import fastf1
import os
import joblib
import requests

# ----------------------------
# Configuration
# ----------------------------
CACHE_DIR = "C:/Users/siddh/OneDrive/Desktop/UNIVERSITY/f1-predictor/f1_cache"
MODEL_PATH = "models/gb_model_canada.pkl"
WEATHER_API_KEY = "ad088f9bbef7fc4b0ebfea6ed606fec2"
YEAR_2024 = 2024
YEAR_2025 = 2025
EVENT_NAME_CANADA = "Canadian Grand Prix"
QUALI_TIME_COL = "QualifyingTime"

fastf1.Cache.enable_cache(CACHE_DIR)

# ----------------------------
# Helper Functions
# ----------------------------

def get_event_id_by_name(year, event_name):
    schedule = fastf1.get_event_schedule(year)
    event_row = schedule[schedule['EventName'] == event_name]
    if not event_row.empty:
        return int(event_row['RoundNumber'].iloc[0])
    else:
        raise ValueError(f"No event found named '{event_name}' in {year}")

def load_race_data(year, event_id):
    session = fastf1.get_session(year, event_id, "R")
    session.load()
    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Stint", "Compound"]].copy()
    laps.dropna(subset=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"], inplace=True)
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()
    return laps

def add_lap_consistency_feature(laps_data):
    lap_std = laps_data.groupby("Driver")["LapTime (s)"].std().reset_index()
    lap_std.rename(columns={"LapTime (s)": "LapTimeStd"}, inplace=True)
    return lap_std

def add_historical_performance(driver):
    historical_scores = {
        "VER": 92, "LEC": 86, "HAM": 91, "NOR": 79, "SAI": 81,
        "RUS": 84, "PIA": 77, "ALO": 88, "STR": 71, "OCO": 75
    }
    return historical_scores.get(driver, 75)

def adjust_qualifying_for_weather(df, rain_prob):
    if rain_prob >= 0.75:
        df["AdjustedQualiTime"] = df["QualifyingTime"] * 1.05
    else:
        df["AdjustedQualiTime"] = df["QualifyingTime"]
    return df

def add_tyre_strategy_impact(session_year, event_id):
    session = fastf1.get_session(session_year, event_id, 'R')
    session.load()
    stints = session.laps[["Driver", "Stint", "Compound"]]
    avg_stints = stints.groupby(["Driver", "Compound"])["Stint"].count().unstack(fill_value=0).add_prefix("Tyre_")
    
    # Ensure all tyre types are included
    for compound in ['HARD', 'MEDIUM', 'SOFT']:
        if f"Tyre_{compound}" not in avg_stints.columns:
            avg_stints[f"Tyre_{compound}"] = 0
    
    return avg_stints.reset_index()

def validate_data_integrity(df):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("⚠️ Warning: Missing values detected:")
        print(missing_values)
    else:
        print("✅ No missing values found.")

def save_predictions_to_csv(results_df, filename="predictions/canada_gp_2025.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results_df.to_csv(filename, index=False)
    print(f"💾 Predictions saved to {filename}")

def engineer_features(laps_data, quali_data, driver_to_team, team_performance_score, avg_position_change, rain_prob, temp):
    sector_times = laps_data.groupby("Driver").agg({
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean"
    }).reset_index()
    sector_times["TotalSectorTime (s)"] = sector_times[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].sum(axis=1)

    lap_std = add_lap_consistency_feature(laps_data)
    tyre_strat = add_tyre_strategy_impact(YEAR_2024, EVENT_ID_CANADA)

    merged = quali_data.merge(sector_times[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
    merged = merged.merge(lap_std, on="Driver", how="left")
    merged = merged.merge(tyre_strat, on="Driver", how="left")

    merged["Team"] = merged["Driver"].map(driver_to_team)
    merged["TeamPerformanceScore"] = merged["Team"].map(team_performance_score)
    merged["AveragePositionChange"] = merged["Driver"].map(avg_position_change)
    merged["RainProbability"] = rain_prob
    merged["Temperature"] = temp
    merged["HistoricalPerformance"] = merged["Driver"].apply(add_historical_performance)

    # --- Ensure Tyre Columns Exist ---
    for col in ["Tyre_HARD", "Tyre_MEDIUM", "Tyre_SOFT"]:
        if col not in merged.columns:
            merged[col] = 0  # Default value if tyre data is missing

    valid_drivers = merged["Driver"].isin(laps_data["Driver"].unique())
    merged = merged[valid_drivers]
    merged = adjust_qualifying_for_weather(merged, rain_prob)
    validate_data_integrity(merged)
    return merged

def prepare_data(merged_data, laps_data):
    X = merged_data[[
        "AdjustedQualiTime", "RainProbability", "Temperature", "TeamPerformanceScore",
        "CleanAirRacePace (s)", "AveragePositionChange", "LapTimeStd", "HistoricalPerformance",
        "Tyre_HARD", "Tyre_MEDIUM", "Tyre_SOFT"
    ]]
    y = laps_data.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)
    return X_imputed, y

def train_and_save_model(X_train, y_train, save_path=MODEL_PATH):
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37)
    model.fit(X_train, y_train)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)
    return model

def get_weather_forecast(lat=45.5017, lon=-73.5663, forecast_time="2025-06-09 19:00:00"):
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return 0, 20
    data = response.json()
    forecast = next((item for item in data["list"] if item["dt_txt"] == forecast_time), None)
    if not forecast:
        return 0, 20
    return forecast["pop"], forecast["main"]["temp"]

# ----------------------------
# Static Data for Canada GP
# ----------------------------

clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600,
    "ALO": 94.784333, "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444,
    "STR": 95.318250, "HUL": 95.345455, "OCO": 95.682128
}

team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51,
    "Ferrari": 114, "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6,
    "Racing Bulls": 10, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari",
    "RUS": "Mercedes", "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin",
    "TSU": "Racing Bulls", "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine",
    "STR": "Aston Martin"
}

average_position_change_canada = {
    "VER": -0.7, "NOR": 1.3, "PIA": 0.2, "RUS": 0.4, "SAI": -0.1,
    "ALB": 0.8, "LEC": -1.6, "OCO": -0.3, "HAM": 0.4, "STR": 1.2,
    "GAS": -0.2, "ALO": -0.4, "HUL": 0.0
}

qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO", "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime": [72.889, 72.675, 73.001, None, 73.212, 73.109, 72.764, 73.301, 73.012, 73.601, 73.221, 73.112, 73.411]
})
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# ----------------------------
# Main Script Logic
# ----------------------------

if __name__ == "__main__":
    EVENT_ID_CANADA = get_event_id_by_name(YEAR_2024, EVENT_NAME_CANADA)
    laps_data = load_race_data(YEAR_2024, EVENT_ID_CANADA)
    rain_prob, temperature = get_weather_forecast()

    merged_data = engineer_features(
        laps_data=laps_data,
        quali_data=qualifying_2025,
        driver_to_team=driver_to_team,
        team_performance_score=team_performance_score,
        avg_position_change=average_position_change_canada,
        rain_prob=rain_prob,
        temp=temperature
    )

    X_imputed, y = prepare_data(merged_data, laps_data)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)

    model = train_and_save_model(X_train, y_train)

    predictions = model.predict(X_imputed)
    merged_data["PredictedRaceTime (s)"] = predictions
    results = merged_data.sort_values("PredictedRaceTime (s)")

    # ----------------------------
    # Streamlit Dashboard Begins
    # ----------------------------

    st.set_page_config(page_title="🏎️ Canadian GP 2025 Prediction", layout="wide")

    st.markdown("""
    <style>
    body {
        background: linear-gradient(to right, #1a1a1a, #2c2c2c);
        color: white;
        font-family: 'Orbitron', sans-serif;
    }
    .podium-box {
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        font-size: 1.2em;
        box-shadow: 0 4px 10px rgba(0,0,0,0.5);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("🏁 2025 Canadian Grand Prix Prediction Dashboard")
    st.subheader("Powered by Machine Learning | Predicted Using Historical Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🥇 Podium Prediction")
        podium = results.head(3)
        cols = st.columns(3)
        for i, (_, row) in enumerate(podium.iterrows()):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
            with cols[i]:
                st.markdown(f"""
                <div class='podium-box' style='background:#ffcc00{';color:black;' if i == 0 else ''}'>
                    <strong>{medal} {row['Driver']}</strong><br>
                    {row['PredictedRaceTime (s)']:.2f}s
                </div>
                """, unsafe_allow_html=True)

        st.markdown("### 📊 Full Predictions")
        st.dataframe(results[["Driver", "PredictedRaceTime (s)"]].reset_index(drop=True))

        st.markdown("### 🧠 Feature Importance")
        fig, ax = plt.subplots()
        features = [
            "AdjustedQualiTime", "RainProbability", "Temperature", "TeamPerformanceScore",
            "CleanAirRacePace (s)", "AveragePositionChange", "LapTimeStd", "HistoricalPerformance",
            "Tyre_HARD", "Tyre_MEDIUM", "Tyre_SOFT"
        ]
        ax.barh(features, model.feature_importances_, color="#e60000")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Contribution to Predictions")
        st.pyplot(fig)

    with col2:
        st.markdown("### 🌤️ Weather Forecast")
        st.info(f"🌧️ Rain Probability: {rain_prob * 100:.1f}%\n\n🌡️ Temperature: {temperature}°C")

        st.markdown("### 🏎️ Live Car Animation")
        st.image("assets/f1_car.gif", width=300)

        st.markdown("### 📈 Model Accuracy")
        mae = mean_absolute_error(y_test, model.predict(X_test))
        st.metric("Model Error (MAE)", f"{mae:.2f} seconds")

        st.markdown("### 📂 Download Predictions")
        st.download_button("💾 Download CSV", data=results.to_csv(index=False), file_name="canada_gp_2025_predictions.csv")
