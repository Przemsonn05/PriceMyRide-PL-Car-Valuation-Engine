import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from urllib.parse import urlencode
from datetime import datetime
import folium
from streamlit_folium import st_folium

CURRENT_YEAR = datetime.now().year

st.set_page_config(
    page_title="CarVal PL - Car Price Prediction",
    page_icon="https://em-content.zobj.net/source/twitter/408/automobile_1f697.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* -- Base -- */
.stApp {
    background: #0a0f1a;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
html, body { scroll-behavior: smooth; }

/* -- Sidebar -- */
[data-testid="stSidebar"] {
    background: linear-gradient(195deg, #0d1321 0%, #101829 50%, #0b1120 100%) !important;
    border-right: 1px solid rgba(56,189,248,0.08);
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    border: 1px solid rgba(56,189,248,0.10);
    color: #64748b;
    border-radius: 10px;
    font-family: 'Inter', sans-serif;
    font-size: 13.5px;
    font-weight: 500;
    transition: all 0.2s ease;
    text-align: left;
    padding: 10px 16px;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(56,189,248,0.07);
    border-color: rgba(56,189,248,0.30);
    color: #e2e8f0;
    transform: translateX(3px);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(14,165,233,0.12) !important;
    border-color: rgba(56,189,248,0.45) !important;
    color: #7dd3fc !important;
    font-weight: 600 !important;
}

/* -- Typography -- */
h1, h2, h3, h4, h5 {
    font-family: 'Inter', sans-serif !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.025em;
}
p, li, span { color: #94a3b8; }

/* -- CTA button -- */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 13px 28px;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.01em;
    transition: all 0.25s ease;
    box-shadow: 0 4px 20px rgba(14,165,233,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(14,165,233,0.40);
}

/* -- Glass cards -- */
.glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 28px 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
    transition: all 0.25s ease;
}
.glass-card:hover {
    background: rgba(255,255,255,0.045);
    border-color: rgba(56,189,248,0.22);
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.25);
}

/* -- Price result card -- */
.price-card {
    background: linear-gradient(145deg, #0c1a2e 0%, #111d35 50%, #0d1525 100%);
    border: 1px solid rgba(56,189,248,0.35);
    border-radius: 20px;
    padding: 48px 36px 40px;
    text-align: center;
    margin: 28px 0 20px;
    box-shadow: 0 0 60px rgba(14,165,233,0.12), inset 0 1px 0 rgba(255,255,255,0.05);
    position: relative; overflow: hidden;
}
.price-card::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(circle, rgba(14,165,233,0.06) 0%, transparent 50%);
    pointer-events: none;
}
.price-label {
    font-family: 'Inter', sans-serif; font-size: 11px; font-weight: 600;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: #38bdf8; margin-bottom: 8px;
}
.price-value {
    font-family: 'Inter', sans-serif;
    font-size: 56px; font-weight: 900;
    color: #f8fafc; letter-spacing: -0.04em;
    line-height: 1; margin: 12px 0 8px;
}
.price-range {
    font-family: 'Inter', sans-serif;
    font-size: 14px; color: #64748b; margin-top: 6px;
}

/* -- Info / Warning boxes -- */
.info-box {
    background: rgba(14,165,233,0.06);
    border: 1px solid rgba(14,165,233,0.20);
    border-left: 3px solid #0ea5e9;
    border-radius: 10px; padding: 16px 20px; margin: 14px 0;
}
.info-box p { color: #bae6fd !important; margin: 0; line-height: 1.7; font-size: 14px; }

.warning-box {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.22);
    border-left: 3px solid #f59e0b;
    border-radius: 10px; padding: 16px 20px; margin: 14px 0;
}
.warning-box p { color: #fde68a !important; margin: 0; line-height: 1.7; font-size: 14px; }

/* -- Intro text above charts -- */
.chart-intro {
    font-size: 14.5px; color: #94a3b8; line-height: 1.7;
    margin: 4px 0 16px; max-width: 900px;
}

/* -- Metric tiles -- */
.metric-tile {
    background: linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 24px 16px; text-align: center;
}
.metric-tile .val {
    font-family: 'Inter', sans-serif;
    font-size: 30px; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #818cf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1;
}
.metric-tile .lbl {
    font-size: 11px; color: #475569; margin-top: 8px;
    text-transform: uppercase; letter-spacing: 0.10em; font-weight: 500;
}

/* -- Form inputs -- */
.stSelectbox label, .stTextInput label, .stSlider label,
.stNumberInput label {
    color: #64748b !important; font-size: 12px !important;
    font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em;
}

/* -- Section divider -- */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 32px 0; }

/* -- Tables / DataFrames -- */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* -- Otomoto CTA -- */
.otomoto-btn {
    display: inline-block;
    background: linear-gradient(135deg, #dc2626, #b91c1c);
    color: white !important; text-decoration: none;
    padding: 12px 26px; border-radius: 10px;
    font-family: 'Inter', sans-serif; font-weight: 700; font-size: 13px;
    margin-top: 12px; transition: all 0.25s;
    box-shadow: 0 4px 16px rgba(220,38,38,0.25); letter-spacing: 0.02em;
}
.otomoto-btn:hover {
    box-shadow: 0 8px 28px rgba(220,38,38,0.40); transform: translateY(-2px);
}

/* -- Hero -- */
.hero-badge {
    display: inline-block;
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(56,189,248,0.28);
    border-radius: 100px; padding: 6px 20px;
    font-size: 12px; color: #38bdf8; font-weight: 600;
    letter-spacing: 0.06em; margin-bottom: 20px;
}

/* -- Feature card -- */
.feature-icon { font-size: 36px; margin-bottom: 10px; display: block; }
.feature-title {
    font-family: 'Inter', sans-serif; font-size: 16px; font-weight: 700;
    color: #f1f5f9 !important; margin-bottom: 8px;
}
.feature-desc { font-size: 13.5px; color: #64748b; line-height: 1.65; }

/* -- Card secondary buttons -- */
div[data-testid="column"] .stButton > button {
    background: rgba(14,165,233,0.08) !important;
    border: 1px solid rgba(14,165,233,0.22) !important;
    color: #38bdf8 !important; font-size: 13px !important;
    font-weight: 600 !important; padding: 9px 18px !important;
    border-radius: 10px !important; box-shadow: none !important;
    letter-spacing: 0.02em; margin-top: -6px;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(14,165,233,0.18) !important;
    border-color: #38bdf8 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(14,165,233,0.15) !important;
}

/* -- Section header -- */
.section-header {
    font-family: 'Inter', sans-serif;
    font-size: 20px; font-weight: 700;
    color: #e2e8f0 !important;
    margin: 28px 0 12px; padding-bottom: 10px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

/* -- Form section headings -- */
.form-section-title {
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 700;
    color: #7dd3fc !important; text-transform: uppercase;
    letter-spacing: 0.08em; margin: 20px 0 14px;
    padding: 10px 14px;
    background: rgba(14,165,233,0.06); border-radius: 8px;
    border-left: 3px solid #0ea5e9;
}

/* -- Business insight card -- */
.biz-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 24px; margin-bottom: 16px;
    border-left: 3px solid #0ea5e9;
}
.biz-card h4 {
    font-size: 16px !important; font-weight: 700 !important;
    margin: 0 0 10px !important; color: #e2e8f0 !important;
}
.biz-card p { font-size: 14px; color: #94a3b8; line-height: 1.7; margin: 0; }
.biz-card .highlight { color: #38bdf8; font-weight: 600; }

/* -- Tabs -- */
.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: rgba(255,255,255,0.02);
    border-radius: 12px; padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; padding: 8px 20px;
    font-family: 'Inter', sans-serif; font-weight: 600; font-size: 13px;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Brand knowledge-base
# ---------------------------------------------------------------------------

BRAND_FREQUENCY_MAP: dict[str, int] = {
    "volkswagen": 22000, "toyota": 15000, "audi": 14000, "bmw": 13500,
    "mercedes-benz": 13000, "opel": 11000, "ford": 10500, "kia": 10000,
    "hyundai": 9500, "renault": 9000, "peugeot": 8000, "skoda": 8000,
    "seat": 7500, "honda": 6500, "volvo": 6000, "mazda": 5500,
    "nissan": 5000, "mitsubishi": 4500, "fiat": 4000, "citroen": 3800,
    "suzuki": 3500, "subaru": 3000, "land rover": 2500, "jeep": 2200,
    "mini": 2000, "alfa romeo": 1800, "lexus": 1500, "infiniti": 800,
    "dacia": 3000, "tesla": 1200, "porsche": 1000, "jaguar": 700,
    "chevrolet": 600, "dodge": 300, "cadillac": 200, "bentley": 80,
    "ferrari": 60, "lamborghini": 50, "rolls-royce": 40, "maserati": 90,
    "mclaren": 30, "aston martin": 35, "lotus": 45, "maybach": 25,
    "lada": 400, "trabant": 150, "polonez": 120, "syrena": 80,
    "warszawa": 60, "wartburg": 70, "gaz": 55, "moskwicz": 65,
}

ULTRA_LUXURY_BRANDS = {
    'ferrari', 'lamborghini', 'rolls-royce', 'bentley', 'mclaren',
    'bugatti', 'koenigsegg', 'pagani', 'aston martin', 'maybach'
}
LUXURY_BRANDS = {
    'mercedes-benz', 'bmw', 'audi', 'porsche', 'lexus', 'jaguar',
    'maserati', 'tesla', 'land rover', 'infiniti', 'lincoln', 'genesis',
    'cadillac', 'volvo'
}
PREMIUM_BRANDS = {
    'alfa romeo', 'mini', 'saab', 'ds automobiles', 'cupra', 'alpine',
    'lotus', 'subaru', 'acura', 'baic', 'ssangyong'
}
MASS_MARKET_BRANDS = {
    'volkswagen', 'toyota', 'ford', 'hyundai', 'kia', 'honda',
    'opel', 'chevrolet', 'peugeot', 'renault', 'seat', 'skoda',
    'fiat', 'nissan', 'mazda', 'mitsubishi', 'suzuki', 'dacia',
    'citroen', 'citroën', 'dodge', 'ram', 'jeep', 'chrysler',
    'lancia', 'daewoo', 'lada'
}
ALL_PREMIUM_SET = ULTRA_LUXURY_BRANDS | LUXURY_BRANDS | PREMIUM_BRANDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_brand_tier(brand: str) -> str:
    b = brand.strip().lower()
    if b in ULTRA_LUXURY_BRANDS: return "Ultra_Luxury"
    if b in LUXURY_BRANDS:       return "Luxury"
    if b in PREMIUM_BRANDS:      return "Premium"
    if b in MASS_MARKET_BRANDS:  return "Mass_Market"
    return "Niche"


def get_age_category(age: float) -> str:
    if age < 3:  return "New"
    if age < 9:  return "Recent"
    if age < 17: return "Used"
    return "Old"


def get_performance_category(hp_per_liter: float) -> str:
    if pd.isna(hp_per_liter): return "Unknown"
    if hp_per_liter < 60:     return "Economy"
    if hp_per_liter < 100:    return "Standard"
    if hp_per_liter < 150:    return "Performance"
    return "High_Performance"


def get_usage_category(mileage_per_year: float) -> str:
    if pd.isna(mileage_per_year): return "Unknown"
    if mileage_per_year < 10000:  return "Low"
    if mileage_per_year < 20000:  return "Average"
    if mileage_per_year < 30000:  return "High"
    return "Very_High"


def generate_otomoto_link(brand, model, year, price_min, price_max):
    base_url = "https://www.otomoto.pl/osobowe"
    brand_clean = brand.lower().replace(' ', '-').replace('o\u0308', 'o').replace('e\u0301', 'e')
    model_clean = model.lower().replace(' ', '-') if model else ""
    params = {
        'search[filter_enum_make]': brand,
        'search[filter_float_year:from]': max(year - 1, 1915),
        'search[filter_float_year:to]': min(year + 1, CURRENT_YEAR),
        'search[filter_float_price:from]': int(price_min * 0.89),
        'search[filter_float_price:to]': int(price_max * 1.10),
    }
    if model_clean:
        return f"{base_url}/{brand_clean}/{model_clean}?{urlencode(params)}"
    return f"{base_url}/{brand_clean}?{urlencode(params)}"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def prepare_input_data(user_inputs: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_inputs])

    brand_lower = df["Vehicle_brand"].iloc[0].strip().lower()
    model_lower = df["Vehicle_model"].iloc[0].strip().lower()

    df["Vehicle_age"] = CURRENT_YEAR - df["Production_year"].astype(int)
    age = int(df["Vehicle_age"].iloc[0])

    feat_str = df["Features"].iloc[0]
    feat_list = [f.strip() for f in str(feat_str).split(",") if f.strip()] if feat_str else []
    df["Num_features"] = len(feat_list)
    df["Features"] = str(feat_list)

    df["Age_category"] = get_age_category(age)
    df["Is_new_car"] = int(age < 3)
    df["Is_old_car"] = int(age > 16)

    mileage = float(df["Mileage_km"].iloc[0])
    power   = float(df["Power_HP"].iloc[0])
    disp    = float(df["Displacement_cm3"].iloc[0])

    mileage_per_year = mileage / max(age, 1)
    df["Mileage_per_year"] = mileage_per_year
    df["Usage_intensity"]  = get_usage_category(mileage_per_year)

    disp_safe = max(disp, 100.0)
    hp_per_liter = power / (disp_safe / 1000.0)
    df["HP_per_liter"]         = hp_per_liter
    df["Performance_category"] = get_performance_category(hp_per_liter)

    is_premium  = int(brand_lower in ALL_PREMIUM_SET)
    is_supercar = int(power > 500 and is_premium)
    is_collector = int(age > 25)

    df["Is_premium"]   = is_premium
    df["Is_supercar"]  = is_supercar
    df["Is_collector"] = is_collector

    df["Listing_year"] = CURRENT_YEAR

    df["Mileage_km_log"]       = np.log1p(max(mileage, 0))
    df["Power_HP_log"]         = np.log1p(max(power, 0))
    df["Displacement_cm3_log"] = np.log1p(max(disp, 0))

    df["Vehicle_age_squared"]  = age ** 2
    df["Power_HP_squared"]     = power ** 2
    df["Mileage_km_squared"]   = mileage ** 2

    df["Age_Mileage_interaction"]  = age * mileage
    df["Power_Age_interaction"]    = power * age
    df["Mileage_per_year_Age"]     = mileage_per_year * age

    brand_freq = BRAND_FREQUENCY_MAP.get(brand_lower, 500)
    brandmodel_freq = max(brand_freq // 5, 10)

    max_freq = 22000.0
    raw_rarity = np.log1p(max_freq / max(brand_freq, 1))
    max_rarity = np.log1p(max_freq / 1.0)
    rarity_index = min(raw_rarity / max_rarity, 1.0)

    df["Brand_tier"]           = get_brand_tier(df["Vehicle_brand"].iloc[0])
    df["Brand_frequency"]      = brand_freq
    df["Rarity_index"]         = round(rarity_index, 4)
    df["BrandModel_frequency"] = brandmodel_freq

    if brand_freq <= 5:     pop = "Ultra_Rare"
    elif brand_freq <= 20:  pop = "Rare"
    elif brand_freq <= 100: pop = "Uncommon"
    elif brand_freq <= 500: pop = "Common"
    else:                   pop = "Popular"
    df["Brand_popularity"] = pop

    df = df.drop(columns=["Production_year", "Drive"], errors="ignore")

    expected_columns = [
        "Condition", "Vehicle_brand", "Vehicle_model",
        "Mileage_km", "Power_HP", "Displacement_cm3",
        "Fuel_type", "Transmission", "Doors_number", "Colour",
        "Origin_country", "First_owner", "Offer_location", "Features",
        "Vehicle_age", "Num_features", "Age_category",
        "Is_new_car", "Is_old_car",
        "Mileage_per_year", "Usage_intensity", "HP_per_liter",
        "Performance_category", "Is_premium", "Is_supercar", "Is_collector",
        "Listing_year",
        "Mileage_km_log", "Power_HP_log", "Displacement_cm3_log",
        "Vehicle_age_squared", "Power_HP_squared", "Mileage_km_squared",
        "Age_Mileage_interaction", "Power_Age_interaction",
        "Mileage_per_year_Age",
        "Brand_tier", "Brand_frequency", "Rarity_index",
        "BrandModel_frequency", "Brand_popularity",
    ]

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    return df[expected_columns]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="Przemsonn/poland-car-price-model",
            filename="final_car_price_model.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model could not be loaded: {e}")
        return None


@st.cache_data
def load_sample_data():
    np.random.seed(42)
    cities = [
        "Warszawa", "Krakow", "Wroclaw", "Poznan", "Gdansk",
        "Szczecin", "Bydgoszcz", "Lublin", "Katowice", "Bialystok",
        "Gdynia", "Czestochowa", "Radom", "Sosnowiec", "Torun",
        "Kielce", "Gliwice", "Zabrze", "Bytom", "Olsztyn",
        "Lodz", "Rzeszow", "Zielona Gora", "Opole", "Gorzow Wielkopolski",
        "Bielsko-Biala", "Plock", "Legnica", "Tarnow", "Chorzow"
    ]
    city_coords = {
        "Warszawa": (52.2297, 21.0122), "Krakow": (50.0647, 19.9450),
        "Wroclaw": (51.1079, 17.0385), "Poznan": (52.4064, 16.9252),
        "Gdansk": (54.3520, 18.6466), "Szczecin": (53.4285, 14.5528),
        "Bydgoszcz": (53.1235, 18.0084), "Lublin": (51.2465, 22.5684),
        "Katowice": (50.2649, 19.0238), "Bialystok": (53.1325, 23.1688),
        "Gdynia": (54.5189, 18.5305), "Czestochowa": (50.8118, 19.1203),
        "Radom": (51.4027, 21.1471), "Sosnowiec": (50.2862, 19.1040),
        "Torun": (53.0138, 18.5984), "Kielce": (50.8661, 20.6286),
        "Gliwice": (50.2945, 18.6714), "Zabrze": (50.3249, 18.7856),
        "Bytom": (50.3483, 18.9160), "Olsztyn": (53.7784, 20.4801),
        "Lodz": (51.7592, 19.4560), "Rzeszow": (50.0413, 22.0010),
        "Zielona Gora": (51.9356, 15.5062), "Opole": (50.6669, 17.9231),
        "Gorzow Wielkopolski": (52.7311, 15.2287), "Bielsko-Biala": (49.8225, 19.0468),
        "Plock": (52.5464, 19.7065), "Legnica": (51.2070, 16.1553),
        "Tarnow": (50.0121, 20.9858), "Chorzow": (50.2975, 18.9448)
    }
    city_sales = {city: np.random.randint(500, 5000) for city in cities}
    return city_sales, city_coords


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

model_data = load_model()
if model_data:
    pipeline = (
        model_data["model_pipeline"]
        if isinstance(model_data, dict)
        else model_data
    )
else:
    st.error("Model cannot be loaded. Check your internet connection.")
    st.stop()


# ---------------------------------------------------------------------------
# Navigation state
# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"


def navigate_to(page: str) -> None:
    st.session_state.page = page
    st.rerun()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(f"""
    <div style='padding: 24px 8px 20px; text-align: center;'>
        <div style='font-size: 28px; margin-bottom: 4px;'>
            <span style='background: linear-gradient(135deg, #38bdf8, #818cf8);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                         font-weight: 900; font-size: 24px; font-family: Inter, sans-serif;'>
                CarVal PL
            </span>
        </div>
        <div style='font-size: 10px; color: #334155; margin-top: 4px;
                    text-transform: uppercase; letter-spacing: 0.14em; font-weight: 500;'>
            Poland Market &middot; {CURRENT_YEAR}
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_items = [
        ("Home",             "home"),
        ("Price Prediction", "predict"),
        ("Regional Market",  "regional"),
        ("Visualizations",   "visualizations"),
        ("About Model",      "info"),
    ]

    for label, key in nav_items:
        btn_type = "primary" if st.session_state.page == key else "secondary"
        if st.button(f"  {label}", use_container_width=True,
                     type=btn_type, key=f"nav_{key}"):
            navigate_to(key)

    st.markdown(
        "<hr style='border-color: rgba(255,255,255,0.04); margin: 20px 0;'>",
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style='text-align: center; padding: 0 8px;'>
        <p style='font-size: 10px; color: #1e293b; line-height: 2; margin: 0; font-weight: 500;'>
            XGBoost &middot; scikit-learn<br>
            200 000+ listings &middot; R&sup2; 93.0%<br>
            <span style='color:#1e293b;'>Przemsonn/poland-car-price-model</span>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers for charts
# ---------------------------------------------------------------------------

PLOT_BG = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#94a3b8",
    title_font_size=16,
    title_font_family="Inter",
    margin=dict(t=48, b=32, l=16, r=16)
)


def _section(title: str) -> None:
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def _intro(text: str) -> None:
    st.markdown(f"<p class='chart-intro'>{text}</p>", unsafe_allow_html=True)


def _insight(text: str) -> None:
    st.markdown(
        f"<div class='info-box'><p>{text}</p></div>",
        unsafe_allow_html=True
    )


def _show_image(path: str) -> None:
    _, img_col, _ = st.columns([0.3, 3.4, 0.3])
    with img_col:
        st.image(path, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Home
# ═══════════════════════════════════════════════════════════════════════════

def home_page() -> None:
    st.markdown(f"""
    <div style='padding: 56px 0 40px; text-align: center;'>
        <div class='hero-badge'>Polish Automotive Market &middot; {CURRENT_YEAR}</div>
        <h1 style='font-size: 60px; font-weight: 900; color: #f1f5f9 !important;
                   letter-spacing: -0.04em; line-height: 1.05; margin: 0 0 20px;'>
            What is your car<br>worth today?
        </h1>
        <p style='font-size: 18px; color: #64748b; max-width: 520px;
                  margin: 0 auto; line-height: 1.7;'>
            AI-powered valuation trained on 200 000+ real listings
            from the Polish used-car market. Instant, transparent, data-driven.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, cta_col, _ = st.columns([1, 1.6, 1])
    with cta_col:
        if st.button("Get your car valuation now", use_container_width=True):
            navigate_to("predict")

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # Feature cards
    c1, c2, c3 = st.columns(3)
    cards = [
        ("Instant Valuation",
         "Enter your car's details and receive a market estimate in seconds, "
         "powered by a gradient-boosting model trained on real transaction data.",
         "predict", "Value my car"),
        ("Regional Analysis",
         "Explore how prices differ between cities and regions across Poland. "
         "See where premium brands concentrate and which areas are most active.",
         "regional", "View map"),
        ("Data & Charts",
         "Interactive visualizations covering price distributions, depreciation "
         "curves, fuel trends, brand rankings, and model diagnostics.",
         "visualizations", "Browse charts"),
    ]
    for col, (title, desc, target, btn_label) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center; min-height: 190px;'>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(btn_label, key=f"card_btn_{target}", use_container_width=True):
                navigate_to(target)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Model performance summary
    st.markdown(
        "<h2 style='text-align: center; margin-bottom: 24px;'>Model Performance</h2>",
        unsafe_allow_html=True
    )
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("93.0%", "R&sup2; Score"),
        ("35 170 PLN", "RMSE"),
        ("11 900 PLN", "MAE"),
        ("18.6%", "MAPE"),
    ]
    for col, (val, lbl) in zip([m1, m2, m3, m4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-tile'>
                <div class='val'>{val}</div>
                <div class='lbl'>{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='max-width: 860px; margin: 0 auto; text-align: center;
                font-size: 15px; line-height: 1.75; color: #64748b;'>
        The XGBoost model explains approximately <b style='color:#38bdf8;'>93%</b> of the
        variance in used-car prices on the Polish market. Most predictions fall within
        &plusmn;19% of the actual transaction price, with an average absolute error of
        around <b style='color:#38bdf8;'>11 900 PLN</b>. Performance is strongest for
        mass-market vehicles (2010-2024, economy-premium segment) and intentionally
        lower for vintage or exotic cars with very few comparable listings.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    # About project (expanded)
    st.markdown(
        "<h2 style='text-align: center; margin-bottom: 24px;'>About the Project</h2>",
        unsafe_allow_html=True
    )
    st.markdown(f"""
    <div style='max-width: 880px; margin: 0 auto; font-size: 15px; line-height: 1.8;
                color: #64748b;'>
        <p>
            <b style='color: #e2e8f0;'>Car Price Prediction Poland</b> is an end-to-end
            machine-learning project designed to estimate used-car prices on the Polish
            automotive market. The model was trained on data collected directly from
            <b style='color: #e2e8f0;'>Otomoto</b> - Poland's largest car-listing
            platform - and achieves an accuracy of approximately
            <b style='color: #38bdf8;'>R&sup2; = 93.0%</b>.
        </p>
        <p>
            The project covers the <b style='color: #e2e8f0;'>complete data science
            lifecycle</b>: automated web scraping with stratified sampling across market
            segments, thorough data cleaning and validation, extensive exploratory data
            analysis, domain-driven feature engineering (41 features derived from 14 raw
            inputs), systematic model selection with cross-validation, Bayesian
            hyperparameter optimisation via Optuna, and production deployment as a
            containerised Streamlit application with a model hosted on Hugging Face Hub.
        </p>
        <p>
            Four model architectures were evaluated during development:
            <b style='color: #e2e8f0;'>Ridge Regression</b> (R&sup2; 72.4%) served as
            a linear baseline; <b style='color: #e2e8f0;'>Random Forest</b> (R&sup2; 92.2%)
            demonstrated the power of ensemble methods;
            <b style='color: #e2e8f0;'>XGBoost Base</b> (R&sup2; 93.0%) was selected as
            the production model for its best overall test-set performance; and
            <b style='color: #e2e8f0;'>XGBoost Weighted</b> (R&sup2; 92.0%) explored
            sample weighting to improve predictions on rare luxury segments.
        </p>
        <p>
            The feature engineering pipeline creates 27 derived features including
            depreciation proxies (vehicle age, age-squared), usage intensity metrics
            (mileage per year), performance ratios (HP per litre), brand-level market
            features (tier classification, rarity index, popularity bins), and
            interaction terms that capture non-linear relationships between age,
            mileage, and engine power. These features were designed using domain
            knowledge of the automotive market and validated through SHAP-based
            explainability analysis.
        </p>
        <p>
            The dataset of <b style='color: #e2e8f0;'>200 000+ listings</b> was
            collected using a custom stratified scraper that ensures balanced
            representation across popular brands, luxury marques, and electric vehicles.
            All prices reflect the {CURRENT_YEAR - 2}-{CURRENT_YEAR} Polish market
            to avoid mixing incompatible price regimes from different economic periods.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tech stack cards
    t1, t2, t3 = st.columns(3)
    tech_cards = [
        ("Machine Learning",
         "XGBoost 2.0 with Optuna Bayesian optimisation (50+ trials), "
         "scikit-learn pipelines, category-encoders for high-cardinality features."),
        ("Data Pipeline",
         "Custom Otomoto scraper with stratified sampling, Pandas for ETL, "
         "NumPy for numerical transformations. 200 000+ listings processed."),
        ("Deployment",
         "Streamlit web application, Docker containerisation, "
         "Hugging Face Hub for model hosting, Plotly & Folium for visualisations."),
    ]
    for col, (title, desc) in zip([t1, t2, t3], tech_cards):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='min-height: 140px;'>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div class='warning-box' style='max-width: 880px; margin: 20px auto 0;'>
        <p><b>Important:</b> The model performs best for mass-market vehicles
        (economy, standard, and premium segments, 2010-2024). Accuracy may be lower for
        <b>luxury vehicles, rare models, vintage cars, and supercars</b> due to
        limited training data for those segments.</p>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Price Prediction
# ═══════════════════════════════════════════════════════════════════════════

def predict_page() -> None:
    st.markdown("""
    <h1 style='margin-bottom: 6px;'>Vehicle Valuation</h1>
    <p style='color: #64748b; margin-bottom: 20px; font-size: 15px;'>
        Fill in the form below - fields marked with <b style='color:#f87171;'>*</b>
        are required for an accurate estimate.
    </p>
    """, unsafe_allow_html=True)

    if st.button("Back to Home"):
        navigate_to("home")

    brand_list = sorted([
        "Abarth", "Acura", "Aixam", "Alfa Romeo", "Alpine", "Aston Martin",
        "Audi", "Austin", "Autobianchi", "Baic", "Bentley", "BMW", "Buick",
        "Cadillac", "Chevrolet", "Chrysler", "Citroen", "Cupra", "Dacia",
        "Daewoo", "Daihatsu", "DFSK", "DKW", "Dodge", "DS Automobiles",
        "FAW", "Ferrari", "Fiat", "Ford", "Gaz", "GMC", "Honda", "Hummer",
        "Hyundai", "Infiniti", "Isuzu", "Iveco", "Jaguar", "Jeep", "Kia",
        "Lada", "Lamborghini", "Lancia", "Land Rover", "Lexus", "Lincoln",
        "Lotus", "Warszawa", "Maserati", "Maybach", "Mazda", "McLaren",
        "Mercedes-Benz", "Mercury", "MG", "Microcar", "MINI", "Mitsubishi",
        "Moskwicz", "Nissan", "NSU", "Nysa", "Oldsmobile", "Opel",
        "Toyota", "Tata", "Uaz", "Zuk", "Trabant", "Suzuki", "Inny",
        "Volvo", "Subaru", "Volkswagen", "SsangYong", "Saab", "Plymouth",
        "Renault", "Peugeot", "Rolls-Royce", "RAM", "Triumph", "Rover",
        "Wolga", "Tarpan", "Polonez", "Pontiac", "Porsche", "Santana",
        "Saturn", "Scion", "Seat", "Skoda", "Smart", "Syrena", "Talbot",
        "Tavria", "Tesla", "Vanderhall", "Vauxhall", "Wartburg",
        "Zaporozec", "Zastava"
    ])

    country_list = [
        "Germany", "USA", "Japan", "France", "Italy", "United Kingdom",
        "South Korea", "China", "Sweden", "Spain", "Netherlands", "Belgium",
        "Canada", "Australia", "India", "Russia", "Mexico", "Brazil",
        "Czech Republic", "Poland", "Turkey", "Austria", "Switzerland",
        "Denmark", "Norway", "Finland", "Portugal", "Greece", "Thailand", "Vietnam"
    ]

    fuel_list     = ["Gasoline", "Gasoline + LPG", "Diesel", "Hybrid",
                     "Gasoline + CNG", "Hydrogen", "Electric"]
    body_list     = ["small_cars", "coupe", "city_cars", "convertible",
                     "compact", "SUV", "sedan", "station_wagon", "minivan"]
    color_list    = ["Black", "White", "Grey", "Silver", "Blue",
                     "Red", "Yellow", "Green", "Other"]
    trans_list    = ["Manual", "Automatic"]

    with st.form("prediction_form"):
        st.markdown(
            "<div class='form-section-title'>Vehicle Identity</div>",
            unsafe_allow_html=True
        )
        col1, col2 = st.columns(2, gap="large")
        with col1:
            brand_choice  = st.selectbox("Brand *", ["-- select --"] + brand_list)
            vehicle_model = st.text_input("Model", placeholder="e.g. Golf, Astra, Corolla...")
            condition     = st.selectbox("Condition *", ["-- select --", "Used", "New"])
        with col2:
            production_year = st.slider("Production Year *", 1915, CURRENT_YEAR, 2015)
            colour          = st.selectbox("Colour", ["-- select --"] + color_list)
            origin_country  = st.selectbox("Country of Origin", ["-- select --"] + country_list)

        st.markdown(
            "<div class='form-section-title'>Engine & Mechanics</div>",
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            mileage_km       = st.slider("Mileage (km) *", 0, 1_000_000, 100_000, step=1000)
            fuel_type        = st.selectbox("Fuel Type *", ["-- select --"] + fuel_list)
        with c2:
            power_HP         = st.slider("Power (HP) *", 10, 1_500, 150)
            transmission     = st.selectbox("Transmission *", ["-- select --"] + trans_list)
        with c3:
            displacement_cm3 = st.slider("Displacement (cm3) *", 0, 8_000, 2_000, step=50)
            car_type         = st.selectbox("Body Type *", ["-- select --"] + body_list)

        st.markdown(
            "<div class='form-section-title'>Additional Information</div>",
            unsafe_allow_html=True
        )
        a1, a2, a3 = st.columns(3, gap="medium")
        with a1:
            first_owner    = st.selectbox("First Owner", ["No", "Yes"])
        with a2:
            doors_number   = st.selectbox("Doors", [3, 4, 5, 2], index=1)
        with a3:
            offer_location = st.text_input("Location", placeholder="e.g. Warszawa, Krakow...")

        features_input = st.text_area(
            "Equipment / Features (comma-separated, optional)",
            placeholder="e.g. ABS, GPS, Leather seats, Climatronic, Parking sensors",
            height=80
        )

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button(
            "Calculate Valuation", use_container_width=True
        )

    if submitted:
        errors = []
        if brand_choice == "-- select --": errors.append("Brand")
        if fuel_type    == "-- select --": errors.append("Fuel Type")
        if transmission == "-- select --": errors.append("Transmission")
        if car_type     == "-- select --": errors.append("Body Type")
        if condition    == "-- select --": errors.append("Condition")

        if errors:
            st.error(f"Please fill in the required fields: **{', '.join(errors)}**")
        else:
            with st.spinner("Calculating valuation..."):
                try:
                    user_inputs = {
                        "Condition":        condition,
                        "Vehicle_brand":    brand_choice,
                        "Vehicle_model":    vehicle_model or "Unknown",
                        "Production_year":  production_year,
                        "Mileage_km":       mileage_km,
                        "Power_HP":         power_HP,
                        "Displacement_cm3": displacement_cm3,
                        "Fuel_type":        fuel_type,
                        "Drive":            "Front wheels",
                        "Transmission":     transmission,
                        "Type":             car_type,
                        "Doors_number":     int(doors_number),
                        "Colour":           colour if colour != "-- select --" else "Other",
                        "Origin_country":   origin_country if origin_country != "-- select --" else "unknown",
                        "First_owner":      1 if first_owner == "Yes" else 0,
                        "Offer_location":   offer_location or "Unknown",
                        "Features":         features_input or "",
                    }

                    final_df = prepare_input_data(user_inputs)
                    y_log  = pipeline.predict(final_df)[0]
                    price  = float(np.expm1(y_log))

                    price_low  = price * 0.85
                    price_high = price * 1.15

                    st.markdown(f"""
                    <div class='price-card'>
                        <div class='price-label'>Estimated Market Value</div>
                        <div class='price-value'>{price:,.0f} PLN</div>
                        <div class='price-range'>
                            Estimated range:&nbsp;
                            <b style='color:#e2e8f0;'>{price_low:,.0f}</b>
                            &nbsp;-&nbsp;
                            <b style='color:#e2e8f0;'>{price_high:,.0f}</b> PLN
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    vehicle_age      = CURRENT_YEAR - production_year
                    mileage_per_year = mileage_km / max(vehicle_age, 1)
                    hp_per_liter     = power_HP / max(displacement_cm3 / 1000, 0.1)
                    brand_tier       = get_brand_tier(brand_choice)

                    d1, d2, d3, d4 = st.columns(4)
                    with d1: st.metric("Vehicle Age",    f"{vehicle_age} years")
                    with d2: st.metric("Annual Mileage", f"{mileage_per_year:,.0f} km")
                    with d3: st.metric("Specific Power", f"{hp_per_liter:.1f} HP/L")
                    with d4: st.metric("Market Segment", brand_tier)

                    otomoto_url = generate_otomoto_link(
                        brand_choice, vehicle_model or "",
                        production_year, price_low, price_high
                    )
                    st.markdown(f"""
                    <div class='info-box' style='margin-top: 24px;'>
                        <p style='margin: 0 0 8px; font-size: 14px;'>
                            Looking for a <b>{brand_choice} {vehicle_model or ''}</b>
                            around <b>{price:,.0f} PLN</b>?
                        </p>
                        <a class='otomoto-btn' href='{otomoto_url}' target='_blank'>
                            Browse similar cars on Otomoto
                        </a>
                    </div>
                    """, unsafe_allow_html=True)

                    if brand_tier in ("Ultra_Luxury", "Niche") or power_HP > 500 or vehicle_age > 30:
                        st.markdown("""
                        <div class='warning-box' style='margin-top: 16px;'>
                            <p><b>Note:</b> This vehicle belongs to a special market segment
                            (luxury / vintage / high-performance). Valuation accuracy may be
                            reduced due to limited comparable listings in the training data.</p>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Valuation error: {e}")
                    st.info("Please verify your inputs and try again. "
                            "If the problem persists, the model service may be temporarily unavailable.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Regional Market
# ═══════════════════════════════════════════════════════════════════════════

def regional_page() -> None:
    st.markdown("""
    <h1 style='margin-bottom: 6px;'>Regional Market</h1>
    <p style='color: #64748b; margin-bottom: 20px; font-size: 15px;'>
        Geographic distribution of car listings across Poland (200 000+ entries).
    </p>
    """, unsafe_allow_html=True)

    if st.button("Back to Home"):
        navigate_to("home")

    st.markdown("""
    <div class='info-box'>
        <p>Illustrative data - geographic distribution of listings from Polish
        automotive platforms. Circle size and colour reflect the number of listings
        in each city.</p>
    </div>
    """, unsafe_allow_html=True)

    city_sales, city_coords = load_sample_data()

    poland_map = folium.Map(
        location=[52.0, 19.0], zoom_start=6, tiles="CartoDB dark_matter"
    )
    max_sales = max(city_sales.values())

    for city, sales in city_sales.items():
        if city in city_coords:
            lat, lon = city_coords[city]
            radius = 10 + (sales / max_sales) * 32
            color = (
                "#ef4444" if sales > 3000
                else "#f97316" if sales > 2000
                else "#eab308" if sales > 1000
                else "#22c55e"
            )
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=f"<b>{city}</b><br>{sales:,} listings",
                color=color, fill=True, fillColor=color,
                fillOpacity=0.72, weight=2
            ).add_to(poland_map)

    st_folium(poland_map, width=None, height=560)

    st.markdown("### Top 10 Cities by Listings")
    sorted_cities = sorted(city_sales.items(), key=lambda x: x[1], reverse=True)[:10]
    chart_data = pd.DataFrame(sorted_cities, columns=["City", "Listings"])

    fig = px.bar(
        chart_data, x="City", y="Listings",
        color="Listings", color_continuous_scale="ice",
        title="Number of Listings by City"
    )
    fig.update_layout(**PLOT_BG, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div class='feature-title'>Urban vs Rural</div>
            <div class='feature-desc'>
                Major cities (Warszawa, Krakow, Wroclaw) account for over 60% of listings.
                Higher vehicle turnover in metropolitan areas drives more competitive pricing.
                Premium brands are more frequently listed in large cities, reflecting higher
                purchasing power in urban centres.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div class='feature-title'>Regional Trends</div>
            <div class='feature-desc'>
                Western Poland shows higher average prices due to proximity to German imports.
                Coastal cities tend to favour imported German vehicles.
                Eastern regions more frequently offer budget-segment brands.
                Southern Poland (Silesia) leads in total number of offers due to
                population density.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def visualizations_page() -> None:
    st.markdown("""
    <h1 style='margin-bottom: 6px;'>Data Visualizations</h1>
    <p style='color: #64748b; margin-bottom: 24px; font-size: 15px;'>
        Key findings from the analysis of 200 000+ car listings scraped from Otomoto.
        Each chart is accompanied by an interpretation and practical takeaways.
    </p>
    """, unsafe_allow_html=True)

    if st.button("Back to Home"):
        navigate_to("home")

    viz_tab, model_tab, biz_tab = st.tabs([
        "EDA Insights", "Model Analysis", "Business Insights"
    ])

    # ==================================================================
    # TAB 1: EDA Insights
    # ==================================================================
    with viz_tab:

        # -- 1. Price Distribution ------------------------------------
        _section("Price Distribution")
        _intro(
            "Understanding how prices are distributed is the first step in any "
            "pricing analysis. The chart below shows the raw price histogram "
            "alongside the log-transformed version used during model training."
        )
        _show_image("images/eda_price_distribution.png")
        _insight(
            "The left panel reveals a strongly right-skewed distribution: the "
            "majority of vehicles are listed below 100 000 PLN, with a median "
            "of approximately 35 500 PLN. The mean is pulled significantly higher "
            "by a small number of premium and luxury vehicles, making the median "
            "a far better measure of the 'typical' market price. "
            "The right panel shows the same data on a logarithmic scale, which "
            "produces a near-normal bell curve. Training on log-transformed prices "
            "stabilises variance, reduces the influence of extreme outliers, and "
            "allows the model to learn proportional (percentage-based) price "
            "changes rather than absolute PLN differences. This is crucial because "
            "a 10 000 PLN error matters far more on a 30 000 PLN car than on a "
            "300 000 PLN luxury vehicle."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -- 2. Depreciation -----------------------------------------
        _section("Value Depreciation Over Time")
        _intro(
            "Depreciation is the single strongest price driver in the used-car market. "
            "The following chart tracks how median prices decline as vehicles age."
        )
        _show_image("images/eda_depreciation_analysis.png")
        _insight(
            "Depreciation is steepest during the first 3 years: a typical car "
            "loses 40-50% of its initial value once it leaves the 'new-car' bracket. "
            "This reflects the loss of the new-car premium as soon as the vehicle "
            "is registered and the rapid accumulation of initial mileage. After this "
            "sharp decline, the rate of depreciation slows considerably, and median "
            "prices stabilise once vehicles reach approximately 10-15 years of age. "
            "Interestingly, very old vehicles (25+ years) can experience a price "
            "increase as they enter the vintage or collector category - such models "
            "attract enthusiasts and niche buyers willing to pay premiums that may "
            "exceed the car's original market value. This non-linear relationship "
            "is why Vehicle_age_squared is one of the model's most important features."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -- 3. Mileage vs Price by Age -------------------------------
        _section("Mileage vs Price by Vehicle Age")
        _intro(
            "Mileage and age interact in complex ways. A low-mileage old car is not the "
            "same as a low-mileage new car. This scatter plot separates vehicles "
            "into four age cohorts to reveal these dynamics."
        )
        _show_image("images/eda_mileage_vs_price_by_age.png")
        _insight(
            "The mileage-price relationship changes dramatically across age groups. "
            "New vehicles (0-2 years) command premiums of 50k-1M PLN even with "
            "some mileage, because they retain dealer-level residual value. "
            "Recent cars (3-8 years) retain strong value below 100k km but depreciate "
            "steeply above that threshold. Used vehicles (9-16 years) dominate the "
            "mass market below 200k PLN - this is where the model achieves its best "
            "accuracy due to abundant training data. Old cars (>16 years) are "
            "generally priced below 50k PLN, with notable exceptions for vintage "
            "collectibles that defy the standard depreciation curve. "
            "This interaction explains why the Age_Mileage_interaction feature "
            "ranks among the top 15 most important features in the final model."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -- 4. Top 20 Brands ----------------------------------------
        _section("Median Price - Top 20 Brands")
        _intro(
            "Brand positioning is a key factor in used-car pricing. This chart ranks "
            "the 20 most popular brands by their median listing price."
        )
        _show_image("images/eda_median_preice_top20_brands.png")
        _insight(
            "Mercedes-Benz leads with a median price of approximately 62 000 PLN, "
            "followed closely by BMW (~60 000 PLN) and Audi (~50 000 PLN). These "
            "three German premium brands form a clear upper tier separated from the "
            "mass-market cluster at 20 000-40 000 PLN. Brands whose median exceeds "
            "the overall market median are highlighted, marking the premium boundary. "
            "The gap between premium and mass-market brands is substantial - nearly "
            "2x in median price - which is why Brand_tier is an effective feature. "
            "Notably, Volkswagen sits at the top of the mass-market range, "
            "reflecting its broad model portfolio spanning from budget Polos to "
            "premium Touaregs. This within-brand variance is captured by the "
            "BrandModel_frequency and Rarity_index features."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -- 5. Fuel Type Trends -------------------------------------
        _section("Price Trends by Fuel Type")
        _intro(
            "Different fuel types follow distinct market trajectories. This chart "
            "shows how average listing prices have evolved across production years "
            "for each powertrain category."
        )
        _show_image("images/eda_average_car_price_over_the_years_by_fuel_type.png")
        _insight(
            "Electric vehicles show a sharp price increase post-2010, reflecting "
            "battery technology maturation and premium positioning - most EVs on the "
            "Polish market are priced above the overall median. Hybrid vehicles show "
            "moderate growth in the mid-range segment, driven largely by Toyota's "
            "hybrid lineup. Diesel and Gasoline follow steady historical increases "
            "with notable divergence post-2018 as diesel faces regulatory headwinds "
            "in European markets. LPG-converted vehicles consistently trade at a "
            "discount, reflecting buyer perception of higher maintenance complexity. "
            "These fuel-type dynamics are captured by the model through the Fuel_type "
            "categorical feature, which carries a SHAP value of ~0.02."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

    # ==================================================================
    # TAB 2: Model Analysis
    # ==================================================================
    with model_tab:

        # -- SHAP Feature Importance ----------------------------------
        _section("SHAP Feature Importance")
        _intro(
            "SHAP (SHapley Additive exPlanations) values measure each feature's "
            "average marginal contribution to predictions. Unlike split-based "
            "importance, SHAP provides a theoretically grounded, model-agnostic "
            "measure of feature impact."
        )
        _show_image("images/SHAP_feature_importance_professional.png")
        _insight(
            "Vehicle Age dominates with a mean SHAP value of ~0.55, confirming "
            "depreciation as the single strongest price driver in the Polish "
            "used-car market. Engine Power (HP) ranks second at ~0.18, reflecting "
            "that performance specifications create the largest intra-age price "
            "variation. Vehicle Model (~0.15) captures brand- and model-specific "
            "residual value that cannot be explained by generic specifications. "
            "Mileage (~0.12) is less influential than age, suggesting that buyers "
            "weight vehicle freshness (age) more heavily than accumulated wear "
            "(mileage). Vehicle Type and Fuel Type carry marginal SHAP values "
            "(0.01-0.03), indicating their effects are largely subsumed by the "
            "more informative primary features."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -- XGBoost Feature Importance -------------------------------
        _section("XGBoost Split-Based Feature Importance")
        _intro(
            "Split-based importance measures how frequently each feature is used "
            "to create decision splits across all trees. It complements SHAP by "
            "showing which features the model relies on most structurally."
        )
        _show_image("images/model3_feature_importance_xgb.png")
        _insight(
            "Is_new_car emerges as the single most critical split feature, "
            "capturing the structural price discontinuity between new and used "
            "vehicles - the first registration event causes an immediate 20-30% "
            "value drop. Vehicle_age_squared (21%) and Vehicle_age (8%) together "
            "account for ~29% of all splits, modelling the non-linear depreciation "
            "curve. Transmission (10%) reflects the substantial automatic gearbox "
            "premium in the Polish market, where manual transmission is still "
            "the default for many brands. Interestingly, Brand_frequency and "
            "Rarity_index appear in the top 20 despite being engineered features, "
            "validating the domain-driven feature engineering approach."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -- Error Analysis -------------------------------------------
        _section("Error Analysis by Vehicle Age")
        _intro(
            "Analysing prediction errors by production year reveals where the "
            "model performs well and where it struggles. Symmetric errors near "
            "zero indicate good calibration."
        )
        _show_image("images/corrected_residuals_vs_year_of_production_xgb_before_cleaning.png")
        _insight(
            "Residuals for mass-market vehicles (2000-2021) are tightly clustered "
            "around zero with low variance, confirming strong model calibration "
            "in this core segment. The largest errors appear for vintage vehicles "
            "(pre-1980, RMSE ~59 000 PLN) and modern supercars, both of which are "
            "priced by rarity and collector demand rather than standard specifications. "
            "Vehicles from 1990-2000 show slightly higher variance than the 2000-2020 "
            "range, likely because this transition period includes both well-maintained "
            "classics and heavily depreciated daily drivers with identical specs. "
            "These patterns confirm the model's warning about reduced accuracy "
            "for luxury, vintage, and exotic segments."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -- Learning Curves ------------------------------------------
        _section("Learning Curves")
        _intro(
            "Learning curves track model performance as training data increases. "
            "The gap between training and validation scores indicates how much "
            "the model benefits from additional data."
        )
        _show_image("images/tuned_model_learning_curves.png")
        _insight(
            "Training and validation curves converge smoothly with a minimal "
            "overfitting gap (< 2% R&sup2; difference at full dataset size). "
            "A clear plateau is visible beyond ~150 000 samples, suggesting that "
            "additional data alone is unlikely to substantially improve performance "
            "without introducing new feature types. This indicates the model has "
            "reached the limit of what 41 tabular features can capture. Potential "
            "gains would likely require NLP-derived features from listing "
            "descriptions, image-based condition assessment, or external data "
            "sources such as vehicle history reports."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

        # -- Model Comparison -----------------------------------------
        _section("Model Comparison")
        _intro(
            "Four architectures were systematically compared using identical "
            "train/test splits and cross-validation to ensure a fair assessment."
        )
        _show_image("images/model_comparison.png")
        _insight(
            "Ridge Regression (R&sup2; 72.4%) serves as a linear baseline, demonstrating "
            "that even a simple model can capture the dominant linear trends "
            "(depreciation, power). Random Forest (R&sup2; 92.2%) shows a dramatic jump, "
            "proving that non-linear feature interactions are essential for accurate "
            "pricing. XGBoost Base (R&sup2; 93.0%) was selected as the production model "
            "because it achieves the best test-set MAPE (18.6%) while maintaining "
            "stable generalisation. XGBoost Weighted (R&sup2; 92.0%) trades ~1% overall "
            "accuracy for improved performance on rare luxury segments - a reasonable "
            "trade-off for some business use cases, but the base model was preferred "
            "for general deployment."
        )

        st.markdown("<hr>", unsafe_allow_html=True)

    # ==================================================================
    # TAB 3: Business Insights
    # ==================================================================
    with biz_tab:

        st.markdown("""
        <div style='margin-bottom: 28px;'>
            <h2 style='margin-bottom: 8px;'>Business Insights</h2>
            <p style='color: #64748b; font-size: 15px; line-height: 1.7;'>
                Actionable conclusions derived from the analysis of 200 000+ car listings
                and the trained pricing model. These insights are relevant for car dealers,
                fleet managers, insurance companies, and individual buyers or sellers.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insight 1: Depreciation timing
        st.markdown("""
        <div class='biz-card'>
            <h4>1. The First 3 Years Are the Most Expensive</h4>
            <p>
                Vehicles lose <span class='highlight'>40-50% of their value</span> within
                the first 3 years of ownership. For a car purchased new at 120 000 PLN,
                this translates to a depreciation cost of 48 000-60 000 PLN - roughly
                16 000-20 000 PLN per year. After year 3, annual depreciation drops to
                approximately 5 000-8 000 PLN. <b>For cost-conscious buyers, purchasing a
                3-4 year old vehicle offers the best value proposition</b>, as it avoids
                the steepest depreciation while retaining modern safety features and
                warranty coverage.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insight 2: Premium brand premium
        st.markdown("""
        <div class='biz-card'>
            <h4>2. The German Premium Tax Is Real - and Quantifiable</h4>
            <p>
                Mercedes-Benz, BMW, and Audi command a
                <span class='highlight'>60-80% median price premium</span> over comparable
                mass-market vehicles of the same age and mileage. This premium persists across
                all age categories but narrows for vehicles older than 15 years, where
                maintenance costs and parts availability become the dominant pricing factor.
                <b>For dealers, stocking 3-8 year old German premium vehicles offers the highest
                margins</b>, but requires careful inspection to avoid high-mileage units that
                have crossed the cost-benefit threshold.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insight 3: Transmission premium
        st.markdown("""
        <div class='biz-card'>
            <h4>3. Automatic Transmission Commands a Consistent Premium</h4>
            <p>
                In the Polish market, where manual transmission remains the default for many
                brands, automatic gearboxes add approximately
                <span class='highlight'>8-15% to the resale value</span> of equivalent vehicles.
                This premium is strongest for SUVs and premium sedans and weakest for city
                cars. <b>Sellers with automatic-transmission vehicles can position their
                asking price at the upper end of comparable listings</b>. The model captures
                this through Transmission as one of the top 10 most important features.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insight 4: Electric vehicles
        st.markdown("""
        <div class='biz-card'>
            <h4>4. Electric Vehicles Hold Value Differently</h4>
            <p>
                Post-2018 EVs show <span class='highlight'>lower depreciation rates</span>
                than comparable ICE vehicles in the first 5 years, primarily due to limited
                supply on the secondary market and growing demand driven by rising fuel costs
                and urban driving restrictions. However, EVs older than 8 years depreciate
                faster due to battery degradation concerns. <b>This creates a narrow sweet
                spot (3-6 years) where used EVs offer the best value for buyers and the
                highest residual value for sellers.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insight 5: Geographic pricing
        st.markdown("""
        <div class='biz-card'>
            <h4>5. Location Matters More Than Most Sellers Realise</h4>
            <p>
                The same vehicle can be listed at
                <span class='highlight'>10-20% different prices</span> depending on the
                city. Western Poland (proximity to Germany) shows higher average prices for
                German brands, while eastern regions favour budget segments. Major cities
                (Warszawa, Krakow, Wroclaw) account for over 60% of all listings, creating
                higher competition but also faster turnover. <b>Sellers in smaller cities
                may benefit from listing on national platforms rather than local ones to
                reach buyers in higher-price urban markets.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insight 6: Mileage threshold
        st.markdown("""
        <div class='biz-card'>
            <h4>6. The 100 000 km Psychological Barrier</h4>
            <p>
                The data shows a <span class='highlight'>noticeable price drop</span> when
                vehicles cross the 100 000 km threshold, particularly for vehicles under
                8 years old. This drop exceeds what the linear mileage-price relationship
                would predict, suggesting a psychological pricing effect. <b>For sellers
                approaching this threshold, listing the vehicle before crossing 100 000 km
                can preserve 5-8% of the asking price.</b> The model accounts for this
                through the Mileage_km_squared polynomial feature.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insight 7: Equipment value
        st.markdown("""
        <div class='biz-card'>
            <h4>7. Equipment Lists Add Measurable Value</h4>
            <p>
                Listings with <span class='highlight'>10+ listed features</span> (e.g. ABS,
                GPS, leather seats, parking sensors, climatronic) sell at approximately
                8-12% higher prices than sparsely described listings of otherwise identical
                vehicles. This is captured by the Num_features count. <b>Sellers should
                invest time in creating comprehensive equipment lists - it is free and directly
                impacts the model's price estimate and, by extension, buyer perception.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Insight 8: Model accuracy scope
        st.markdown("""
        <div class='biz-card'>
            <h4>8. When to Trust (and Distrust) the Model</h4>
            <p>
                The model achieves its <span class='highlight'>highest accuracy
                (MAPE < 15%)</span> for mass-market vehicles aged 3-15 years with
                30 000-200 000 km. Accuracy degrades significantly for: ultra-luxury
                brands (Ferrari, Lamborghini - MAPE > 40%), vintage collectors (pre-1985),
                rare models with fewer than 20 comparable listings, and highly customised
                vehicles. <b>For business use, the model is best suited as a pricing
                baseline for the mass-market segment, complemented by expert
                appraisal for special vehicles.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: About Model
# ═══════════════════════════════════════════════════════════════════════════

def info_page() -> None:
    st.markdown("""
    <h1 style='margin-bottom: 6px;'>How the Model Works</h1>
    <p style='color: #64748b; margin-bottom: 24px; font-size: 15px;'>
        Architecture, training process, and limitations of the XGBoost pipeline.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
### Model Overview
The application uses an **XGBoost** gradient-boosting model trained on data
scraped from the Polish automotive market to estimate **used car prices** in PLN.
The model processes 41 engineered features and outputs a log-scale prediction
that is inverse-transformed to the original PLN scale.

---

### Why XGBoost?

| Model | R^2 | MAPE | Status |
|-------|-----|------|--------|
| Ridge Regression | 0.724 | 28.5% | Baseline |
| Random Forest | 0.922 | 22.8% | Good |
| **XGBoost (base)** | **0.930** | **18.6%** | Selected |

**Key advantages:**
- Sequential boosting - each tree corrects the previous one's errors
- Robust to noise and missing values
- Inference < 1 second
- L1 / L2 regularisation prevents overfitting on sparse segments

---

### Feature Set (41 features)

**Raw inputs (14):** Brand, Model, Year, Mileage, Power, Displacement, Fuel type,
Transmission, Body type, Doors, Colour, Location, Condition, Country of origin

**Engineered features (27):**

| Feature | Description |
|---------|-------------|
| `Vehicle_age` | Current year - production year |
| `Age_category` | New / Recent / Used / Old |
| `Is_new_car`, `Is_old_car` | Binary flags for age extremes |
| `Mileage_per_year` | Mileage / age |
| `Usage_intensity` | Low / Average / High / Very_High categorical bin |
| `HP_per_liter` | Power per litre of displacement |
| `Performance_category` | Economy / Standard / Performance / High |
| `Is_premium`, `Is_supercar` | Brand-tier binary flags |
| `Is_collector` | Flag for cars > 25 years old |
| `Listing_year` | Year the listing was published |
| `Num_features` | Count of listed equipment items |
| `Brand_tier` | Mass_Market / Premium / Luxury / Ultra_Luxury / Niche |
| `Brand_frequency` | Brand's share of total listings |
| `Rarity_index` | Log-normalised inverse of brand frequency |
| `BrandModel_frequency` | Brand x Model combination frequency |
| `Brand_popularity` | Ultra_Rare / Rare / Uncommon / Common / Popular |
| Log-transformed | `Mileage_km_log`, `Power_HP_log`, `Displacement_cm3_log` |
| Polynomial | `*_squared` terms for age, power, mileage |
| Interaction | `Age x Mileage`, `Power x Age`, `MileagePerYear x Age` |

---

### Final Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R^2** | 93.0% | Explains ~93% of price variance |
| **RMSE** | 35 170 PLN | Root mean squared error |
| **MAE** | 11 900 PLN | Mean absolute error |
| **MAPE** | 18.6% | ~19% average relative deviation |

---

### Training Pipeline

1. **Data collection** - 200 000+ listings scraped from Otomoto (2024-2026)
2. **Preprocessing** - duplicate removal, currency conversion, outlier handling
3. **Feature engineering** - 41 features derived from 14 raw inputs
4. **Model selection** - Ridge vs Random Forest vs XGBoost comparison
5. **Hyperparameter tuning** - Optuna Bayesian optimisation, 50+ trials
6. **Validation** - 80/20 train/test split + 3-fold cross-validation
7. **Deployment** - Streamlit, Hugging Face Hub, Docker

---

### Limitations

**Best accuracy for:**
- Mass-market brands (VW, Toyota, Ford, Opel, Skoda, Kia, Hyundai)
- Standard configurations: 100-300 HP, 50-200k km, 2010-2024

**Reduced accuracy for:**
- Luxury / exotic brands (Ferrari, Lamborghini, Rolls-Royce)
- Vintage cars (> 30 years old)
- Rare models (< 20 examples in the dataset)
- Custom factory configurations not captured by standard features

---

### Tech Stack

- **ML:** XGBoost 2.0, scikit-learn, category-encoders
- **Optimisation:** Optuna (TPE Sampler, Bayesian search)
- **Data:** Pandas, NumPy, scraped from Otomoto
- **Deployment:** Streamlit >= 1.43, Hugging Face Hub, Docker
- **Visualisations:** Plotly, Folium, Matplotlib / Seaborn
""")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

page = st.session_state.page
if   page == "home":           home_page()
elif page == "predict":        predict_page()
elif page == "regional":       regional_page()
elif page == "visualizations": visualizations_page()
elif page == "info":           info_page()
