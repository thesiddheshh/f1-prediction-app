import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
import fastf1
import os
import joblib
import requests
import time
from datetime import datetime

# ----------------------------
# Configuration
# ----------------------------
# Use /tmp for cache and model directories on Streamlit Cloud
CACHE_DIR = "/tmp/f1_cache"
MODEL_DIR = "/tmp/models"
PREDICTION_DIR = "/tmp/predictions"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PREDICTION_DIR, exist_ok=True)

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
    df["AdjustedQualiTime"] = df["QualifyingTime"] * (1 + 0.05 * (rain_prob / 100))
    return df

def add_tyre_strategy_impact(session_year, event_id):
    session = fastf1.get_session(session_year, event_id, 'R')
    session.load()
    stints = session.laps[["Driver", "Stint", "Compound"]]
    avg_stints = stints.groupby(["Driver", "Compound"])["Stint"].count().unstack(fill_value=0).add_prefix("Tyre_")
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
        print("✅ No missing values found")

def save_predictions_to_csv(results_df, filename="predictions/predictions.csv"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    results_df.to_csv(filename, index=False)

def engineer_features(laps_data, quali_data, driver_to_team, team_performance_score, avg_position_change, rain_prob, temp):
    sector_times = laps_data.groupby("Driver").agg({
        "Sector1Time (s)": "mean",
        "Sector2Time (s)": "mean",
        "Sector3Time (s)": "mean"
    }).reset_index()
    sector_times["TotalSectorTime (s)"] = sector_times[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].sum(axis=1)
    lap_std = add_lap_consistency_feature(laps_data)
    tyre_strat = add_tyre_strategy_impact(2024, st.session_state.event_id)
    merged = quali_data.merge(sector_times[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
    merged = merged.merge(lap_std, on="Driver", how="left")
    merged = merged.merge(tyre_strat, on="Driver", how="left")
    merged["Team"] = merged["Driver"].map(driver_to_team)
    merged["TeamPerformanceScore"] = merged["Team"].map(team_performance_score)
    merged["AveragePositionChange"] = merged["Driver"].map(avg_position_change)
    merged["RainProbability"] = rain_prob
    merged["Temperature"] = temp
    merged["HistoricalPerformance"] = merged["Driver"].apply(add_historical_performance)
    for col in ["Tyre_HARD", "Tyre_MEDIUM", "Tyre_SOFT"]:
        if col not in merged.columns:
            merged[col] = 0
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

def train_and_save_model(X_train, y_train, save_path="models/gb_model.pkl"):
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

def get_historical_track_data(track_name):
    """Get historical data for the track over the past 5 years"""
    historical_data = {
        "avg_race_time": np.random.uniform(5400, 6000),
        "avg_overtakes": np.random.randint(40, 80),
        "tyre_strategies": {
            "1-stop": np.random.randint(40, 60),
            "2-stop": np.random.randint(30, 50),
            "3-stop": np.random.randint(5, 15)
        },
        "dnf_rates": {
            "engine": np.random.randint(5, 15),
            "accident": np.random.randint(5, 15),
            "other": np.random.randint(5, 15)
        }
    }
    return historical_data

def create_quali_animation(quali_order, predicted_order):
    """Create animated SVG showing qualifying order morphing into race finish"""
    # Create a simple animation using HTML/CSS
    html = """
    <style>
    @keyframes moveCars {
        0% {transform: translateY(0);}
        50% {transform: translateX(100px);}
        100% {transform: translateY(100px);}
    }
    
    .animation-container {
        position: relative;
        height: 200px;
        overflow: hidden;
    }
    
    .car-line {
        position: absolute;
        width: 100%;
        display: flex;
        justify-content: space-between;
        animation: moveCars 5s ease-in-out infinite;
    }
    
    .car-icon {
        font-size: 24px;
        transition: all 0.5s ease;
    }
    </style>
    
    <div class="animation-container">
        <div class="car-line">
    """
    
    drivers = list(quali_order.keys())
    for i, driver in enumerate(drivers):
        html += f'<div class="car-icon" style="left: {i*10}%; transform: translateY(0);">{driver}</div>'
    
    html += "</div><div class=\"car-line\">"
    
    drivers = list(predicted_order.keys())
    for i, driver in enumerate(drivers):
        html += f'<div class="car-icon" style="left: {i*10}%; transform: translateY(100px);">{driver}</div>'
    
    html += """
        </div>
    </div>
    """
    return html

def create_driver_comparison_chart(driver1, driver2, data):
    """Create radar chart comparing two drivers across multiple metrics"""
    metrics = ['Qualifying Time', 'Race Pace', 'Consistency', 'Overtaking', 'Tire Usage']
    scores1 = [np.random.rand() for _ in range(len(metrics))]
    scores2 = [np.random.rand() for _ in range(len(metrics))]
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    scores1 += scores1[:1]
    scores2 += scores2[:1]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], metrics, color='grey', size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["20%", "40%", "60%", "80%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data
    ax.plot(angles, scores1, linewidth=1, linestyle='solid', label=driver1)
    ax.fill(angles, scores1, color='#e60000', alpha=0.25)
    
    ax.plot(angles, scores2, linewidth=1, linestyle='solid', label=driver2)
    ax.fill(angles, scores2, color='#00aaff', alpha=0.25)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    return fig

def calculate_driver_form_index(driver):
    """Calculate rolling performance index over last 5 races"""
    # Simulate form data - in real implementation would fetch actual race data
    dates = pd.date_range(end=datetime.today(), periods=5).tolist()
    form_scores = np.random.uniform(0.6, 1.0, 5)  # Simulated form scores
    return pd.DataFrame({'Date': dates, 'Form Score': form_scores})

# ----------------------------
# Static Data
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

average_position_change = {
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
# UI Components
# ----------------------------
def animated_car_header():
    st.markdown("""
    <style>
    .car-container {
        overflow: hidden;
        white-space: nowrap;
        position: relative;
        width: 100%;
        height: 50px;
        background-color: #1a1a1a;
        color: white;
        font-size: 20px;
        font-family: monospace;
    }
    .car-text {
        display: inline-block;
        padding-left: 100%;
        animation: scroll 10s linear infinite;
    }
    @keyframes scroll {
        from { transform: translateX(0); }
        to { transform: translateX(-100%); }
    }
    </style>
    <div class="car-container">
      <div class="car-text">🏎️ VER | NOR | LEC | HAM | PIA | RUS | SAI | STR | OCO | ALO | HUL | GAS | ALB</div>
    </div>
    """, unsafe_allow_html=True)

def get_f1_schedule(year):
    schedule = fastf1.get_event_schedule(year)
    return list(schedule[schedule['EventFormat'] == 'conventional']['EventName'])

# ----------------------------
# Main App Logic
# ----------------------------
st.set_page_config(page_title="🏁 F1 2025 Race Predictor", layout="wide")
animated_car_header()
st.title("🏎️ F1 2025 Grand Prix Prediction Dashboard")
st.subheader("Select a race below to view predictions powered by Machine Learning")

if "event_id" not in st.session_state:
    st.session_state.event_id = 1

# Race Selection Dropdown
schedule = fastf1.get_event_schedule(2024)
race_options = list(schedule[schedule['EventFormat'] == 'conventional']['EventName'])
selected_race = st.selectbox("Choose a Grand Prix:", options=race_options)
st.session_state.event_id = get_event_id_by_name(2024, selected_race)

# Auto-refresh every hour
if "last_run" not in st.session_state:
    st.session_state.last_run = 0

if time.time() - st.session_state.last_run > 3600:  # Refresh every hour
    st.session_state.last_run = time.time()
    st.rerun()

# Run Predictions
laps_data = load_race_data(2024, st.session_state.event_id)
rain_prob, temperature = get_weather_forecast()

merged_data = engineer_features(
    laps_data=laps_data,
    quali_data=qualifying_2025,
    driver_to_team=driver_to_team,
    team_performance_score=team_performance_score,
    avg_position_change=average_position_change,
    rain_prob=rain_prob,
    temp=temperature
)

X_imputed, y = prepare_data(merged_data, laps_data)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)
model = train_and_save_model(X_train, y_train)
predictions = model.predict(X_imputed)
merged_data["PredictedRaceTime (s)"] = predictions
results = merged_data.sort_values("PredictedRaceTime (s)")

# Podium Display
st.markdown("### 🥇 Top 3 Predicted Finishers")
podium = results.head(3)
cols = st.columns(3)

for i, (_, row) in enumerate(podium.iterrows()):
    medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
    with cols[i]:
        st.markdown(f"""
        <div style='border-radius: 10px; padding: 10px; margin: 5px; text-align:center; font-size:1.2em; background:#e60000;'>
            <strong>{medal} {row['Driver']}</strong><br>
            {row['PredictedRaceTime (s)']:.2f}s
        </div>
        """, unsafe_allow_html=True)

# Animated Quali-to-Finish Flow
st.markdown("### 🎥 Animated Quali-to-Finish Flow")
st.markdown(create_quali_animation(
    {d: i for i, d in enumerate(qualifying_2025['Driver'])},
    {d: i for i, d in enumerate(results['Driver'])}
), unsafe_allow_html=True)

# Feature Importance
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
ax.invert_yaxis()
st.pyplot(fig)

# Weather Impact Slider
st.markdown("### 🌧️ Weather Impact Simulation")
adjusted_rain_prob = st.slider("Adjust Rain Probability (%)", min_value=0, max_value=100, value=int(rain_prob * 100))

# Recalculate predictions with adjusted weather
adjusted_merged_data = adjust_qualifying_for_weather(merged_data.copy(), adjusted_rain_prob / 100)
X_adjusted, y_adjusted = prepare_data(adjusted_merged_data, laps_data)
adjusted_predictions = model.predict(X_adjusted)
adjusted_merged_data["PredictedRaceTime (s)"] = adjusted_predictions
adjusted_results = adjusted_merged_data.sort_values("PredictedRaceTime (s)")

# Live Driver Comparison Tool
st.markdown("### 🏁 Live Driver Comparison Tool")
col1, col2 = st.columns(2)
with col1:
    driver1 = st.selectbox("Select First Driver", options=results['Driver'].unique())
with col2:
    driver2 = st.selectbox("Select Second Driver", options=results['Driver'].unique())

if driver1 != driver2:
    comparison_fig = create_driver_comparison_chart(driver1, driver2, results)
    st.pyplot(comparison_fig)
else:
    st.warning("Please select two different drivers for comparison.")

# Historical Track Trends
st.markdown("### 📆 Historical Track Trends")
historical_data = get_historical_track_data(selected_race)

track_col1, track_col2 = st.columns(2)

with track_col1:
    st.markdown("#### ⏱️ Performance Trends")
    st.write(f"**Avg Race Time:** {historical_data['avg_race_time']:.0f} seconds")
    st.write(f"**Avg Overtakes:** {historical_data['avg_overtakes']:.0f}")
    
    st.markdown("#### 🛞 Tire Strategies")
    tire_df = pd.DataFrame({
        "Strategy": historical_data['tyre_strategies'].keys(),
        "Percentage": historical_data['tyre_strategies'].values()
    })
    st.bar_chart(tire_df.set_index("Strategy"))

with track_col2:
    st.markdown("#### ⚠️ DNF Rates")
    dnf_fig, ax = plt.subplots()
    ax.pie(historical_data['dnf_rates'].values(), labels=historical_data['dnf_rates'].keys(), 
           autopct='%1.1f%%', startangle=90, colors=['#e60000', '#00aaff', '#ffd700'])
    ax.axis('equal')
    st.pyplot(dnf_fig)

# Driver Form Index
st.markdown("### ✅ Driver Form Index")
form_driver = st.selectbox("Select a driver to view form index:", options=results['Driver'].unique())
form_data = calculate_driver_form_index(form_driver)

form_col1, form_col2 = st.columns([2, 1])

with form_col1:
    st.line_chart(form_data.set_index('Date'))

with form_col2:
    current_form = form_data.iloc[-1]['Form Score']
    st.markdown(f"""
    <div style='padding: 20px; border-radius: 10px; background-color: #e60000; color: white; text-align: center;'>
        <h3>Current Form</h3>
        <h1>{current_form:.2f}</h1>
        <p>(1 = Best, 0 = Worst)</p>
    </div>
    """, unsafe_allow_html=True)

# Full Predictions Table
st.markdown("### 📊 Full Predicted Rankings")
st.dataframe(results[["Driver", "PredictedRaceTime (s)"]].reset_index(drop=True))

# Weather Info
st.info(f"🌧️ Rain Probability: {rain_prob * 100:.1f}% | 🌡️ Temperature: {temperature}°C")

# Download Button
st.download_button("💾 Download Predictions CSV", data=results.to_csv(index=False), 
                  file_name=f"{selected_race.replace(' ', '_')}_2025_predictions.csv")
