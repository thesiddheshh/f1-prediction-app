def load_driver_profile(driver):
    # Load telemetry stats, past performance, etc.
    return {
        "Best Lap": np.random.uniform(70, 80),
        "Avg Lap": np.random.uniform(71, 81),
        "Win Rate (%)": np.random.randint(10, 80),
        "Top 3 Rate (%)": np.random.randint(30, 90)
    }

def show_driver_profile(name):
    st.markdown(f"## 🏎️ Profile: {name}")
    stats = load_driver_profile(name)
    for k, v in stats.items():
        st.metric(k, round(v, 2))