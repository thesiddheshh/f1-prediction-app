import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import os
import joblib

# ----------------------------
# Configuration
# ----------------------------
CACHE_DIR = "C:/Users/siddh/OneDrive/Desktop/UNIVERSITY/f1-predictor/f1_cache"
MODEL_PATH = "models/gb_model_spain.pkl"
WEATHER_API_KEY = "ad088f9bbef7fc4b0ebfea6ed606fec2"  # New API Key
YEAR_2024 = 2024
YEAR_2025 = 2025
EVENT_NAME_SPAIN = "Spanish Grand Prix"
QUALI_TIME_COL = "QualifyingTime"

fastf1.Cache.enable_cache(CACHE_DIR)

# ----------------------------
# Helper Functions
# ----------------------------

def get_weather_forecast(lat=41.5737, lon=2.2586, forecast_time="2025-05-19 13:00:00"):
    """Barcelona Circuit de Catalunya coordinates"""
    url = f"http://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch weather data")
        return 0, 25
    data = response.json()
    forecast = next((item for item in data["list"] if item["dt_txt"] == forecast_time), None)
    if not forecast:
        return 0, 25
    return forecast["pop"], forecast["main"]["temp"]

def load_race_data(year, event_id):
    session = fastf1.get_session(year, event_id, "R")
    session.load()
    laps = session.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time", "Stint", "Compound"]].copy()
    laps.dropna(subset=["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"], inplace=True)
    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()
    return laps

def get_event_id_by_name(year, event_name):
    schedule = fastf1.get_event_schedule(year)
    event_row = schedule[schedule['EventName'] == event_name]
    if not event_row.empty:
        return int(event_row['RoundNumber'].iloc[0])
    else:
        raise ValueError(f"No event found named '{event_name}' in {year}")

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
    return avg_stints.reset_index()

def validate_data_integrity(df):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print("⚠️ Warning: Missing values detected:")
        print(missing_values)
    else:
        print("✅ No missing values found.")

def save_predictions_to_csv(results_df, filename="predictions/spain_gp_2025.csv"):
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
    tyre_strat = add_tyre_strategy_impact(YEAR_2024, EVENT_ID_SPAIN)

    merged = quali_data.merge(sector_times[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
    merged = merged.merge(lap_std, on="Driver", how="left")
    merged = merged.merge(tyre_strat, on="Driver", how="left")

    merged["Team"] = merged["Driver"].map(driver_to_team)
    merged["TeamPerformanceScore"] = merged["Team"].map(team_performance_score)
    merged["AveragePositionChange"] = merged["Driver"].map(avg_position_change)
    merged["RainProbability"] = rain_prob
    merged["Temperature"] = temp
    merged["HistoricalPerformance"] = merged["Driver"].apply(add_historical_performance)

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

def load_model(path=MODEL_PATH):
    return joblib.load(path)

def cross_validate_model(model, X, y):
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f"📊 Cross-validated MAE: {-scores.mean():.2f} ± {scores.std():.2f}")

# ----------------------------
# Static Data
# ----------------------------

clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128
}

team_points = {
    "McLaren": 279, "Mercedes": 147, "Red Bull": 131, "Williams": 51, "Ferrari": 114,
    "Haas": 20, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 10, "Alpine": 7
}
max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "VER": "Red Bull", "NOR": "McLaren", "PIA": "McLaren", "LEC": "Ferrari", "RUS": "Mercedes",
    "HAM": "Mercedes", "GAS": "Alpine", "ALO": "Aston Martin", "TSU": "Racing Bulls",
    "SAI": "Ferrari", "HUL": "Kick Sauber", "OCO": "Alpine", "STR": "Aston Martin"
}

average_position_change_spain = {
    "VER": -0.8, "NOR": 1.2, "PIA": 0.1, "RUS": 0.4, "SAI": -0.2,
    "ALB": 0.9, "LEC": -1.4, "OCO": -0.1, "HAM": 0.5, "STR": 1.0,
    "GAS": -0.3, "ALO": -0.5, "HUL": 0.0
}

qualifying_2025 = pd.DataFrame({
    "Driver": ["VER", "NOR", "PIA", "RUS", "SAI", "ALB", "LEC", "OCO", "HAM", "STR", "GAS", "ALO", "HUL"],
    "QualifyingTime": [72.112, 71.945, 72.432, None, 73.001, 72.789, 72.003, 72.884, 72.567, 73.201, 72.943, 72.612, 73.109]
})
qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

# ----------------------------
# Main Script Logic
# ----------------------------

if __name__ == "__main__":
    EVENT_ID_SPAIN = get_event_id_by_name(YEAR_2024, EVENT_NAME_SPAIN)
    laps_data = load_race_data(YEAR_2024, EVENT_ID_SPAIN)
    rain_prob, temperature = get_weather_forecast()

    merged_data = engineer_features(
        laps_data=laps_data,
        quali_data=qualifying_2025,
        driver_to_team=driver_to_team,
        team_performance_score=team_performance_score,
        avg_position_change=average_position_change_spain,
        rain_prob=rain_prob,
        temp=temperature
    )

    X_imputed, y = prepare_data(merged_data, laps_data)
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)

    # Always retrain to match current feature set
    model = train_and_save_model(X_train, y_train)

    predictions = model.predict(X_imputed)
    merged_data["PredictedRaceTime (s)"] = predictions
    results = merged_data.sort_values("PredictedRaceTime (s)")

    print("\n🏁 Predicted 2025 Spanish GP Winner 🏁\n")
    print(results[["Driver", "PredictedRaceTime (s)"]])

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\n📉 Model Error (MAE): {mae:.2f} seconds")
    cross_validate_model(model, X_imputed, y)

    feature_importance = model.feature_importances_
    features = [
        "AdjustedQualiTime", "RainProbability", "Temperature", "TeamPerformanceScore",
        "CleanAirRacePace (s)", "AveragePositionChange", "LapTimeStd", "HistoricalPerformance",
        "Tyre_HARD", "Tyre_MEDIUM", "Tyre_SOFT"
    ]
    plt.figure(figsize=(8, 5))
    plt.barh(features, feature_importance, color='skyblue')
    plt.xlabel("Importance")
    plt.title("Feature Importance in Race Time Prediction")
    plt.tight_layout()
    plt.show()

    podium = results.iloc[:3][["Driver", "PredictedRaceTime (s)"]]
    print("\n🏆 Predicted in the Top 3 🏆")
    print(f"🥇 P1: {podium.iloc[0]['Driver']}")
    print(f"🥈 P2: {podium.iloc[1]['Driver']}")
    print(f"🥉 P3: {podium.iloc[2]['Driver']}")

    save_predictions_to_csv(results)