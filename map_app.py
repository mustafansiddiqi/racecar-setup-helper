import streamlit as st
import configparser
import os
import matplotlib.pyplot as plt
import numpy as np
import gpxpy
import requests
from matplotlib.collections import LineCollection
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import time

# UI CSS

st.markdown("""
<style>

    /* ---------------------------------------------------
       FONT IMPORTS
    --------------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600&display=swap');

    /* ---------------------------------------------------
       GLOBAL APP BACKGROUND (Dark Navy + Texture)
    --------------------------------------------------- */
    .stApp {
        background-color: #0A1224;
        background-image: url('https://www.transparenttextures.com/patterns/asfalt-light.png');
        color: #E6E8EC;
        font-family: 'Inter', sans-serif;
    }

    /* ---------------------------------------------------
       HEADERS
    --------------------------------------------------- */
    h1, h2, h3, h4, h5 {
        color: #F5F7FA !important;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.5px;
    }

    /* ---------------------------------------------------
       GENERAL TEXT
    --------------------------------------------------- */
    p, span, label, .stMarkdown, .stText, .stCaption {
        color: #E6E8EC !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 15px;
    }

    /* ---------------------------------------------------
       SIDEBAR
    --------------------------------------------------- */
    [data-testid="stSidebar"] {
        background-color: #0D1630 !important;
        color: #E6E8EC !important;
        border-right: 1px solid #1E2A4A;
    }

    [data-testid="stSidebar"] * {
        color: #ECEFF4 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ---------------------------------------------------
       INPUT FIELDS (TEXT + NUMBER)
    --------------------------------------------------- */
    .stTextInput input,
    .stNumberInput input {
        background-color: #162038 !important;
        border: 1px solid #2B3A55 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ---------------------------------------------------
       SELECTBOX — NAVY BOX + WHITE TEXT
    --------------------------------------------------- */

    /* Visible selectbox container (works for all selectboxes, incl. "Select track") */
    .stSelectbox > div > div {
    background-color: #162038 !important;
    border: 1px solid #2B3A55 !important;
    border-radius: 6px !important;

    /* NEW: Increase vertical size */
    padding: 10px 12px !important;     /* more vertical padding */
    min-height: 45px !important;       /* ensure box is tall enough */

    display: flex !important;
    align-items: center !important;    /* vertically center text */
}

    /* Text & arrow inside the selectbox */
    .stSelectbox > div > div * {
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Dropdown menu panel */
    .stSelectbox [role="listbox"] {
        background-color: #0D1630 !important;
        border: 1px solid #2B3A55 !important;
        border-radius: 6px !important;
    }

    /* Dropdown options */
    .stSelectbox [role="option"] {
        background-color: #0D1630 !important;
        color: #FFFFFF !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stSelectbox [role="option"]:hover {
        background-color: #182644 !important;
    }

    /* ---------------------------------------------------
       BUTTONS
    --------------------------------------------------- */
    .stButton>button {
        background-color: #1A2A49 !important;
        color: #FFFFFF !important;
        border: 1px solid #2F456B !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.8rem !important;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.5px;
        transition: 0.15s;
    }

    .stButton>button:hover {
        background-color: #253861 !important;
        border-color: #3C5582 !important;
    }

    /* ---------------------------------------------------
       DOWNLOAD BUTTON
    --------------------------------------------------- */
    .stDownloadButton > button {
        background-color: #1A2A49 !important;
        color: #FFFFFF !important;
        border: 1px solid #2F456B !important;
        border-radius: 6px !important;
        padding: 0.4rem 0.8rem !important;
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.5px;
        transition: 0.15s;
    }

    .stDownloadButton > button:hover {
        background-color: #253861 !important;
        border-color: #3C5582 !important;
    }

    /* ---------------------------------------------------
       EXPANDERS
    --------------------------------------------------- */
    .streamlit-expanderHeader {
        background-color: #162038 !important;
        color: #FFFFFF !important;
        border-radius: 4px !important;
    }

</style>
""", unsafe_allow_html=True)


# File Paths

DATA_DIR = "/Users/Mustafa_1/Mustafa_Mac/US_Life/UChicago/Year 1/Student Employment/RA-Ship/RAship_Code/Rally_Tool/nurburgring_data"
FILES = {
    "sections": os.path.join(DATA_DIR, "sections.ini"),
    "surfaces": os.path.join(DATA_DIR, "surfaces.ini"),
    "gpx": os.path.join(DATA_DIR, "Nordschleife.gpx"),
}

# Nürburgring approximate coordinates for weather API
NURBURGRING_LAT = 50.3356
NURBURGRING_LON = 6.9476


# Utilities – INI/GPX

def read_ini(filepath):
    cfg = configparser.ConfigParser()
    cfg.optionxform = str
    cfg.read(filepath)
    return cfg

def parse_sections(cfg):
    sections = []
    for s in cfg.sections():
        if s.startswith("SECTION_"):
            block = cfg[s]
            sections.append({
                "name": block.get("TEXT", ""),
                "in": float(block.get("IN", 0)),   # meters along track
                "out": float(block.get("OUT", 0))
            })
    return sections

def load_gpx(path):
    with open(path, "r") as f:
        gpx = gpxpy.parse(f)

    lat, lon, ele = [], [], []
    for track in gpx.tracks:
        for seg in track.segments:
            for p in seg.points:
                lat.append(p.latitude)
                lon.append(p.longitude)
                ele.append(p.elevation)

    return np.array(lat), np.array(lon), np.array(ele)

def latlon_to_xy(lat, lon):
    R = 6371000  # meters
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)
    x = (lon - lon0) * np.cos(np.radians(lat0)) * (np.pi / 180) * R
    y = (lat - lat0) * (np.pi / 180) * R
    return x, y

def track_distance(x, y):
    seg = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    seg = np.insert(seg, 0, 0)
    return np.cumsum(seg)


# Track visualization – elevation-colored line

def plot_track_colored_by_elevation(x, y, ele, sections, dist):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Build colored line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colormap = "viridis"

    lc = LineCollection(segments, cmap=colormap)
    lc.set_array(ele)
    lc.set_linewidth(4)
    ax.add_collection(lc)
    plt.colorbar(lc, ax=ax, label="Elevation (m)")

    # Section highlights (slightly thicker)
    for sec in sections:
        mask = (dist >= sec["in"]) & (dist <= sec["out"])
        if mask.sum() == 0:
            continue
        sec_x = x[mask]
        sec_y = y[mask]
        sec_ele = ele[mask]
        pts = np.array([sec_x, sec_y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        sec_lc = LineCollection(segs, cmap=colormap)
        sec_lc.set_array(sec_ele)
        sec_lc.set_linewidth(6)
        ax.add_collection(sec_lc)

    ax.set_title("Nürburgring – Elevation-Colored GPX Track")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.autoscale()
    st.pyplot(fig)

def plot_altitude(ele, dist):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dist / 1000, ele, color="green")
    ax.set_title("Altitude Profile Along Nordschleife")
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Elevation (m)")
    st.pyplot(fig)


# Track statistics for setup logic

def compute_track_stats(ele, dist):
    # gradient in % between points
    d_dist = np.diff(dist)
    d_ele = np.diff(ele)
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient = (d_ele / d_dist) * 100
    gradient[np.isnan(gradient)] = 0
    gradient[np.isinf(gradient)] = 0

    stats = {
        "elev_min": float(np.min(ele)),
        "elev_max": float(np.max(ele)),
        "elev_range": float(np.max(ele) - np.min(ele)),
        "max_uphill_pct": float(np.max(gradient)),
        "max_downhill_pct": float(np.min(gradient)),
        "pct_steep_uphill": float(np.mean(gradient > 5) * 100),
        "pct_steep_downhill": float(np.mean(gradient < -5) * 100),
        "avg_gradient": float(np.mean(gradient)),
    }
    return stats


# OpenWeather API helper (hard-coded key)

OPENWEATHER_API_KEY = "48b8cf776845b1b3b76e183c60826568"

def get_openweather_api_key():
    return OPENWEATHER_API_KEY

def fetch_weather(lat=NURBURGRING_LAT, lon=NURBURGRING_LON):
    api_key = get_openweather_api_key()

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": api_key,
        "units": "metric",
    }

    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.error(f"Error fetching weather: {e}")
        return None

    weather = {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "wind_speed": data["wind"]["speed"],
        "conditions": data["weather"][0]["main"],  # e.g. 'Clear', 'Rain', 'Clouds'
    }
    return weather


# Setup recommendation engine (Option A + braking/damping/etc.)

def recommend_setup(car, track_name, track_stats, weather):
    #  baselines for a generic car
    setup = {
        # Alignment
        "front_camber": -2.5,
        "rear_camber": -2.0,
        "caster": 6.5,
        "front_toe": 0.05,   # toe out (deg)
        "rear_toe": 0.10,    # toe in (deg)

        # Springs (relative to a notional base rate)
        "front_spring_rate_Nmm": 160, 
        "rear_spring_rate_Nmm": 140,

        # Ride height (mm)
        "front_ride_height": 70,
        "rear_ride_height": 80,

        # Dampers (1–10 style)
        "front_bump": 6,
        "front_rebound": 7,
        "rear_bump": 6,
        "rear_rebound": 7,

        # Roll bars (1–10)
        "front_arb": 6,
        "rear_arb": 5,

        # Brakes
        "brake_bias": 0.62,      # fraction front
        "brake_pressure": 0.90,  # 0–1

        # Diff (1–10 style locks & preload)
        "diff_preload": 5,
        "diff_power_lock": 6,
        "diff_coast_lock": 4,

        # Aero (simple)
        "wing_angle": 5,  # 0–10: 0 = minimum, 10 = max
        "rake_mm": 10,    # rear - front

        # Tires
        "tire_compound": "Medium",
        "tire_pressure_front": 28,
        "tire_pressure_rear": 28,

        # Gearing
        "final_drive": "Stock",
        "gearing_note": "Slightly shorten 3rd–5th for better acceleration out of medium-speed corners.",

        "notes": [],
    }

    weight = car["weight"]
    drivetrain = car["drivetrain"]
    power = car["power"]
    base_susp = car["base_susp"]

    elev_range = track_stats["elev_range"]
    steep_up = track_stats["pct_steep_uphill"]
    steep_down = track_stats["pct_steep_downhill"]

    temp = weather["temp"] if weather else 20
    conditions = (weather["conditions"] if weather else "Clear").lower()
    wind = weather["wind_speed"] if weather else 2

    #  Base-suspension feel adjustments 
    if base_susp == "Soft":
        setup["front_spring_rate_Nmm"] -= 10
        setup["rear_spring_rate_Nmm"] -= 10
        setup["front_bump"] -= 1
        setup["rear_bump"] -= 1
        setup["front_rebound"] -= 1
        setup["rear_rebound"] -= 1
        setup["notes"].append("Base car is soft → keep springs/dampers slightly on the softer side.")
    elif base_susp == "Stiff":
        setup["front_spring_rate_Nmm"] += 10
        setup["rear_spring_rate_Nmm"] += 10
        setup["front_bump"] += 1
        setup["rear_bump"] += 1
        setup["front_rebound"] += 1
        setup["rear_rebound"] += 1
        setup["notes"].append("Base car is stiff → keep springs/dampers slightly firmer.")

    #  Track-based adjustments (Nordschleife heuristics) 
    if "nürburgring" in track_name.lower():
        # Nordschleife: bumpy, high-speed, big compressions
        if elev_range > 250:
            setup["front_ride_height"] += 10
            setup["rear_ride_height"] += 10
            setup["front_spring_rate_Nmm"] -= 10
            setup["rear_spring_rate_Nmm"] -= 10
            setup["front_bump"] -= 1
            setup["rear_bump"] -= 1
            setup["notes"].append("Nordschleife elevation/compressions → slightly higher ride height, softer springs and bump damping.")

        # High-speed sections: moderate wing, stable braking
        setup["wing_angle"] = 5
        setup["front_arb"] = 6
        setup["rear_arb"] = 5
        setup["brake_bias"] = 0.61
        setup["notes"].append("High-speed track → moderate aero, stable front-biased braking, medium ARB stiffness.")

    if steep_down > 15:
        setup["brake_bias"] += 0.01
        setup["rear_toe"] += 0.02
        setup["notes"].append("Steep downhill sections → slightly more front brake bias and rear toe-in for stability.")

    #  Weather-based adjustments 
    if "rain" in conditions or "drizzle" in conditions:
        # Wet
        setup["front_camber"] += 0.5
        setup["rear_camber"] += 0.5
        setup["front_spring_rate_Nmm"] -= 10
        setup["rear_spring_rate_Nmm"] -= 10
        setup["front_bump"] -= 1
        setup["rear_bump"] -= 1
        setup["front_rebound"] -= 1
        setup["rear_rebound"] -= 1
        setup["front_toe"] = 0.02
        setup["rear_toe"] = 0.14
        setup["tire_compound"] = "Soft"
        setup["tire_pressure_front"] -= 1
        setup["tire_pressure_rear"] -= 1
        setup["brake_pressure"] = 0.85
        setup["notes"].append("Wet conditions → less camber, softer springs/damping, lower pressures, and gentle braking.")

    elif temp > 28:
        # Hot
        setup["front_camber"] -= 0.3
        setup["rear_camber"] -= 0.2
        setup["front_spring_rate_Nmm"] += 10
        setup["rear_spring_rate_Nmm"] += 10
        setup["front_bump"] += 1
        setup["rear_bump"] += 1
        setup["tire_compound"] = "Hard"
        setup["tire_pressure_front"] += 1
        setup["tire_pressure_rear"] += 1
        setup["notes"].append("Hot track → more negative camber, slightly stiffer springs/damping, harder compound, a bit more pressure.")

    elif temp < 10:
        # Cold
        setup["front_camber"] += 0.2
        setup["rear_camber"] += 0.2
        setup["front_spring_rate_Nmm"] -= 5
        setup["rear_spring_rate_Nmm"] -= 5
        setup["tire_compound"] = "Soft"
        setup["tire_pressure_front"] -= 1
        setup["tire_pressure_rear"] -= 1
        setup["notes"].append("Cold track → reduce camber slightly, softer suspension, softer compound and lower tire pressures for warmup.")

    if wind > 8:
        setup["caster"] += 0.5
        setup["notes"].append("Strong wind → slightly more caster for straight-line stability.")

    #  Car-based adjustments 
    if drivetrain == "RWD":
        setup["rear_toe"] += 0.05
        setup["rear_arb"] += 1
        setup["diff_power_lock"] += 1
        setup["notes"].append("RWD → extra rear toe-in, slightly stiffer rear bar, more diff power lock for traction.")
    elif drivetrain == "FWD":
        setup["front_toe"] += 0.03
        setup["front_arb"] += 1
        setup["notes"].append("FWD → extra front toe-out and stiffer front bar for sharper turn-in.")
    elif drivetrain == "AWD":
        if "rain" in conditions:
            setup["notes"].append("AWD in wet → prioritize traction; leave diff balance neutral.")
        else:
            setup["diff_preload"] += 1
            setup["diff_power_lock"] += 1
            setup["notes"].append("AWD in dry → slightly more diff preload/power lock for better drive off corners.")

    if weight > 1500:
        setup["front_spring_rate_Nmm"] += 10
        setup["rear_spring_rate_Nmm"] += 10
        setup["brake_pressure"] = min(setup["brake_pressure"] + 0.05, 1.0)
        setup["notes"].append("Heavier car → slightly stiffer springs and higher brake pressure.")

    if power > 400 and drivetrain in ["RWD", "AWD"]:
        setup["rear_toe"] += 0.03
        setup["wing_angle"] += 1
        setup["diff_preload"] += 1
        setup["notes"].append("High power → more rear toe-in, a bit more wing, and more diff preload for traction.")

    #  Simple gearing suggestion 
    if power < 300:
        setup["final_drive"] = "Shorter"
        setup["gearing_note"] = "Use a slightly shorter final drive to keep the engine in the power band on climbs."
    elif power > 500:
        setup["final_drive"] = "Stock or slightly longer"
        setup["gearing_note"] = "Keep stock or slightly longer final drive to avoid hitting the limiter on long straights."

    return setup


# PDF export

def generate_setup_pdf(car, track_name, weather, setup):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    x_margin = 50
    y = height - 50

    def line(text, indent=0):
        nonlocal y
        c.drawString(x_margin + indent, y, text)
        y -= 14
        if y < 50:
            c.showPage()
            y_new = height - 50
            y = y_new

    # Header
    c.setFont("Helvetica-Bold", 16)
    line(f"Race Setup Sheet – {track_name}")
    c.setFont("Helvetica", 11)
    line(f"Car: {car['name']}  |  Drivetrain: {car['drivetrain']}  |  Weight: {car['weight']} kg  |  Power: {car['power']} hp")
    line("")

    # Weather
    if weather:
        line("Weather:", 0)
        line(f"- Conditions: {weather['conditions']}", 15)
        line(f"- Temp: {weather['temp']} °C", 15)
        line(f"- Humidity: {weather['humidity']}%", 15)
        line(f"- Wind: {weather['wind_speed']} m/s", 15)
        line("")
    else:
        line("Weather: Not fetched (assumed dry, ~20 °C).")

    # Alignment
    line("Alignment:", 0)
    line(f"- Front camber: {setup['front_camber']:.2f}°", 15)
    line(f"- Rear camber: {setup['rear_camber']:.2f}°", 15)
    line(f"- Caster: {setup['caster']:.2f}°", 15)
    line(f"- Front toe: {setup['front_toe']:.2f}° toe-out", 15)
    line(f"- Rear toe: {setup['rear_toe']:.2f}° toe-in", 15)
    line("")

    # Suspension
    line("Suspension:", 0)
    line(f"- Front ride height: {setup['front_ride_height']:.0f} mm", 15)
    line(f"- Rear ride height: {setup['rear_ride_height']:.0f} mm (rake {setup['rake_mm']} mm)", 15)
    line(f"- Front spring rate: {setup['front_spring_rate_Nmm']} N/mm", 15)
    line(f"- Rear spring rate: {setup['rear_spring_rate_Nmm']} N/mm", 15)
    line(f"- Front dampers: bump {setup['front_bump']}/10, rebound {setup['front_rebound']}/10", 15)
    line(f"- Rear dampers: bump {setup['rear_bump']}/10, rebound {setup['rear_rebound']}/10", 15)
    line(f"- Anti-roll bars: front {setup['front_arb']}/10, rear {setup['rear_arb']}/10", 15)
    line("")

    # Brakes
    line("Brakes:", 0)
    line(f"- Brake bias: {setup['brake_bias']*100:.1f}% front", 15)
    line(f"- Brake pressure: {setup['brake_pressure']*100:.0f}%", 15)
    line("")

    # Diff
    line("Differential:", 0)
    line(f"- Preload: {setup['diff_preload']}/10", 15)
    line(f"- Power lock: {setup['diff_power_lock']}/10", 15)
    line(f"- Coast lock: {setup['diff_coast_lock']}/10", 15)
    line("")

    # Aero
    line("Aero:", 0)
    line(f"- Wing angle: {setup['wing_angle']}/10", 15)
    line(f"- Rake: {setup['rake_mm']} mm (rear higher than front)", 15)
    line("")

    # Tires
    line("Tires:", 0)
    line(f"- Compound: {setup['tire_compound']}", 15)
    line(f"- Front pressure: {setup['tire_pressure_front']} psi", 15)
    line(f"- Rear pressure: {setup['tire_pressure_rear']} psi", 15)
    line("")

    # Gearing
    line("Gearing:", 0)
    line(f"- Final drive: {setup['final_drive']}", 15)
    line(f"- Note: {setup['gearing_note']}", 15)
    line("")

    # Notes / reasoning
    if setup["notes"]:
        line("Notes / Reasoning:", 0)
        for note in setup["notes"]:
            line(f"- {note}", 15)

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


# 8. Streamlit UI

def main():
    st.title("Race Setup Helper")

    #  Track selection 
    track_name = st.selectbox("Select track", ["Nürburgring Nordschleife"])
    
    #  Load track data 
    sec_cfg = read_ini(FILES["sections"])
    sections = parse_sections(sec_cfg)
    lat, lon, ele = load_gpx(FILES["gpx"])
    x, y = latlon_to_xy(lat, lon)
    dist = track_distance(x, y)
    track_stats = compute_track_stats(ele, dist)

    #  Sidebar info 
    st.sidebar.header("Track Info")
    st.sidebar.write(f"Track: {track_name}")
    st.sidebar.write(f"GPX points: {len(lat)}")
    st.sidebar.write(f"Elevation range: {track_stats['elev_range']:.1f} m")
    st.sidebar.write(f"Max uphill: {track_stats['max_uphill_pct']:.1f} %")
    st.sidebar.write(f"Max downhill: {track_stats['max_downhill_pct']:.1f} %")

    #  Car input 
    st.subheader("Car Info")
    col1, col2 = st.columns(2)
    with col1:
        car_name = st.text_input("Car name", "Mustafa's GT Car")
        drivetrain = st.selectbox("Drivetrain", ["RWD", "FWD", "AWD"])
        weight = st.number_input("Weight (kg)", min_value=800, max_value=2000, value=1350, step=10)
    with col2:
        power = st.number_input("Power (hp)", min_value=100, max_value=1000, value=450, step=10)
        base_susp = st.selectbox("Base suspension feel", ["Medium", "Soft", "Stiff"])

    car = {
        "name": car_name,
        "drivetrain": drivetrain,
        "weight": weight,
        "power": power,
        "base_susp": base_susp,
    }

   #  Visualizations 
    st.subheader("Track Map & Elevation")
    plot_track_colored_by_elevation(x, y, ele, sections, dist)

    st.subheader("Altitude Profile")
    plot_altitude(ele, dist)

    #  Weather fetch 
    st.subheader("Weather at Nürburgring")
    weather = None
    if st.button("Fetch current weather from OpenWeather"):
        weather = fetch_weather()

    if weather:
        st.write(
            f"**Conditions:** {weather['conditions']} | "
            f"**Temp:** {weather['temp']} °C | "
            f"**Humidity:** {weather['humidity']}% | "
            f"**Wind:** {weather['wind_speed']} m/s"
        )
    else:
        st.info("Using default dry / ~20°C assumptions if no weather is fetched.")

    #  Setup recommendation 
    st.subheader("Recommended Setup")
    setup = None
    if st.button("Compute setup recommendation"):
        with st.spinner("Computing optimal setup…"):
            time.sleep(2)
        setup = recommend_setup(car, track_name, track_stats, weather)

        st.markdown(f"### For: **{car_name}** on {track_name}")
        colA, colB = st.columns(2)
        with colA:
            st.write("**Alignment**")
            st.write(f"- Front camber: **{setup['front_camber']:.2f}°**")
            st.write(f"- Rear camber: **{setup['rear_camber']:.2f}°**")
            st.write(f"- Caster: **{setup['caster']:.2f}°**")
            st.write(f"- Front toe: **{setup['front_toe']:.2f}° toe-out**")
            st.write(f"- Rear toe: **{setup['rear_toe']:.2f}° toe-in**")

            st.write("**Brakes**")
            st.write(f"- Brake bias: **{setup['brake_bias']*100:.1f}% front**")
            st.write(f"- Brake pressure: **{setup['brake_pressure']*100:.0f}%**")

            st.write("**Tires**")
            st.write(f"- Compound: **{setup['tire_compound']}**")
            st.write(f"- Front pressure: **{setup['tire_pressure_front']} psi**")
            st.write(f"- Rear pressure: **{setup['tire_pressure_rear']} psi**")

        with colB:
            st.write("**Suspension & Damping**")
            st.write(f"- Front ride height: **{setup['front_ride_height']:.0f} mm**")
            st.write(f"- Rear ride height: **{setup['rear_ride_height']:.0f} mm** (rake {setup['rake_mm']} mm)")
            st.write(f"- Front springs: **{setup['front_spring_rate_Nmm']} N/mm**")
            st.write(f"- Rear springs: **{setup['rear_spring_rate_Nmm']} N/mm**")
            st.write(f"- Front dampers: bump **{setup['front_bump']}/10**, rebound **{setup['front_rebound']}/10**")
            st.write(f"- Rear dampers: bump **{setup['rear_bump']}/10**, rebound **{setup['rear_rebound']}/10**")
            st.write(f"- ARBs: front **{setup['front_arb']}/10**, rear **{setup['rear_arb']}/10**")

            st.write("**Diff & Aero & Gearing**")
            st.write(f"- Diff preload: **{setup['diff_preload']}/10**")
            st.write(f"- Diff power lock: **{setup['diff_power_lock']}/10**")
            st.write(f"- Diff coast lock: **{setup['diff_coast_lock']}/10**")
            st.write(f"- Wing angle: **{setup['wing_angle']}/10**")
            st.write(f"- Final drive: **{setup['final_drive']}**")
            st.write(f"- Gearing: {setup['gearing_note']}")

        if setup["notes"]:
            st.write("**Reasoning / Notes:**")
            for note in setup["notes"]:
                st.write(f"- {note}")
        else:
            st.write("_No major adjustments from baseline recommended._")

        # PDF export button
        pdf_buffer = generate_setup_pdf(car, track_name, weather, setup)
        st.download_button(
            label=" Download setup as PDF",
            data=pdf_buffer,
            file_name=f"{car_name.replace(' ', '_')}_{track_name.replace(' ', '_')}_setup.pdf",
            mime="application/pdf"
        )

    st.caption("Prototype race car setup helper.")

if __name__ == "__main__":
    main()
    