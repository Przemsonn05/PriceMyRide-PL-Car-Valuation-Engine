import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from urllib.parse import urlencode
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Car Price Prediction – Poland",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Background and global ── */
.stApp {
    background: #0a0e1a;
    font-family: 'DM Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1628 !important;
    border-right: 1px solid rgba(99,179,237,0.15);
}

[data-testid="stSidebar"] .stButton > button {
    background: transparent;
    border: 1px solid rgba(99,179,237,0.2);
    color: #a0aec0;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-size: 14px;
    font-weight: 500;
    transition: all 0.25s ease;
    text-align: left;
    padding: 10px 16px;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(99,179,237,0.1);
    border-color: #63b3ed;
    color: white;
    transform: translateX(4px);
}

/* ── General text ── */
h1, h2, h3, h4, h5 {
    font-family: 'Syne', sans-serif !important;
    color: #f7fafc !important;
    letter-spacing: -0.02em;
}

p, li, span {
    color: #cbd5e0;
}

/* ── Main CTA button ── */
.stButton > button {
    background: linear-gradient(135deg, #3182ce 0%, #805ad5 100%);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 14px 32px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 16px;
    letter-spacing: 0.02em;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(49,130,206,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(49,130,206,0.5);
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(12px);
    transition: all 0.3s ease;
}

.glass-card:hover {
    background: rgba(255,255,255,0.07);
    border-color: rgba(99,179,237,0.3);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}

/* ── Price card ── */
.price-card {
    background: linear-gradient(135deg, #1a365d 0%, #2d3748 50%, #1a202c 100%);
    border: 1px solid rgba(99,179,237,0.4);
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    margin: 32px 0;
    box-shadow: 0 0 60px rgba(49,130,206,0.2), inset 0 1px 0 rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
}

.price-card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(49,130,206,0.08) 0%, transparent 60%);
    pointer-events: none;
}

.price-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #63b3ed;
    margin-bottom: 12px;
}

.price-value {
    font-family: 'Syne', sans-serif;
    font-size: 56px;
    font-weight: 800;
    color: #f7fafc;
    letter-spacing: -0.03em;
    line-height: 1;
    margin: 16px 0;
}

.price-range {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    color: #a0aec0;
    margin-top: 12px;
}

/* ── Info boxes ── */
.info-box {
    background: rgba(49,130,206,0.1);
    border: 1px solid rgba(49,130,206,0.3);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
}

.info-box p { color: #bee3f8 !important; margin: 0; }

.warning-box {
    background: rgba(214,158,46,0.1);
    border: 1px solid rgba(214,158,46,0.4);
    border-radius: 12px;
    padding: 16px 20px;
    margin: 16px 0;
}

.warning-box p { color: #fbd38d !important; margin: 0; }

/* ── Metrics ── */
.metric-tile {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
}

.metric-tile .val {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #63b3ed;
    line-height: 1;
}

.metric-tile .lbl {
    font-size: 13px;
    color: #718096;
    margin-top: 8px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* ── Form inputs ── */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stSlider {
    border-radius: 10px;
}

/* Fix label readability in forms */
[data-testid="stForm"] label,
.stSelectbox label,
.stTextInput label,
.stSlider label {
    color: #a0aec0 !important;
    font-size: 13px !important;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 32px 0;
}

/* ── Tables ── */
.stDataFrame {
    border-radius: 12px;
    overflow: hidden;
}

/* ── Link button ── */
.otomoto-btn {
    display: inline-block;
    background: linear-gradient(135deg, #e53e3e, #c53030);
    color: white !important;
    text-decoration: none;
    padding: 14px 32px;
    border-radius: 12px;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 15px;
    margin-top: 16px;
    transition: all 0.3s;
    box-shadow: 0 4px 20px rgba(229,62,62,0.3);
    letter-spacing: 0.02em;
}

.otomoto-btn:hover {
    box-shadow: 0 8px 30px rgba(229,62,62,0.5);
    transform: translateY(-2px);
}

/* ── Feature section on home ── */
.feature-icon {
    font-size: 40px;
    margin-bottom: 12px;
    display: block;
}

.feature-title {
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: #f7fafc !important;
    margin-bottom: 8px;
}

.feature-desc {
    font-size: 14px;
    color: #718096;
    line-height: 1.6;
}

/* ── Hero section ── */
.hero-badge {
    display: inline-block;
    background: rgba(49,130,206,0.15);
    border: 1px solid rgba(49,130,206,0.4);
    border-radius: 100px;
    padding: 6px 20px;
    font-size: 13px;
    color: #63b3ed;
    font-weight: 500;
    letter-spacing: 0.05em;
    margin-bottom: 20px;
}

/* ── Card buttons (secondary) ── */
[data-testid="stButton"][aria-label*="card_btn"] > button,
div[data-testid="column"] .stButton > button {
    background: rgba(49,130,206,0.12) !important;
    border: 1px solid rgba(49,130,206,0.3) !important;
    color: #63b3ed !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    border-radius: 10px !important;
    box-shadow: none !important;
    letter-spacing: 0.04em;
    margin-top: -8px;
}

div[data-testid="column"] .stButton > button:hover {
    background: rgba(49,130,206,0.25) !important;
    border-color: #63b3ed !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(49,130,206,0.2) !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id="Przemsonn/poland-car-price-model",
            filename="final_car_price_model.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model was not loaded: {e}")
        return None

@st.cache_data
def load_sample_data():
    np.random.seed(42)
    cities = [
        'Warszawa', 'Kraków', 'Wrocław', 'Poznań', 'Gdańsk',
        'Szczecin', 'Bydgoszcz', 'Lublin', 'Katowice', 'Białystok',
        'Gdynia', 'Częstochowa', 'Radom', 'Sosnowiec', 'Toruń',
        'Kielce', 'Gliwice', 'Zabrze', 'Bytom', 'Olsztyn',
        'Łódź', 'Rzeszów', 'Zielona Góra', 'Opole', 'Gorzów Wielkopolski',
        'Bielsko-Biała', 'Płock', 'Legnica', 'Tarnów', 'Chorzów'
    ]

    city_coords = {
        'Warszawa': (52.2297, 21.0122), 'Kraków': (50.0647, 19.9450),
        'Wrocław': (51.1079, 17.0385), 'Poznań': (52.4064, 16.9252),
        'Gdańsk': (54.3520, 18.6466), 'Szczecin': (53.4285, 14.5528),
        'Bydgoszcz': (53.1235, 18.0084), 'Lublin': (51.2465, 22.5684),
        'Katowice': (50.2649, 19.0238), 'Białystok': (53.1325, 23.1688),
        'Gdynia': (54.5189, 18.5305), 'Częstochowa': (50.8118, 19.1203),
        'Radom': (51.4027, 21.1471), 'Sosnowiec': (50.2862, 19.1040),
        'Toruń': (53.0138, 18.5984), 'Kielce': (50.8661, 20.6286),
        'Gliwice': (50.2945, 18.6714), 'Zabrze': (50.3249, 18.7856),
        'Bytom': (50.3483, 18.9160), 'Olsztyn': (53.7784, 20.4801),
        'Łódź': (51.7592, 19.4560), 'Rzeszów': (50.0413, 22.0010),
        'Zielona Góra': (51.9356, 15.5062), 'Opole': (50.6669, 17.9231),
        'Gorzów Wielkopolski': (52.7311, 15.2287), 'Bielsko-Biała': (49.8225, 19.0468),
        'Płock': (52.5464, 19.7065), 'Legnica': (51.2070, 16.1553),
        'Tarnów': (50.0121, 20.9858), 'Chorzów': (50.2975, 18.9448)
    }

    city_sales = {city: np.random.randint(500, 5000) for city in cities}
    return city_sales, city_coords

def get_brand_reliability_category(brand):
    brand_lower = str(brand).lower()
    luxury = ['ferrari', 'lamborghini', 'rolls-royce', 'bentley', 'aston martin',
              'mclaren', 'maserati', 'porsche']
    american = ['ram', 'dodge', 'chevrolet', 'hummer', 'cadillac']
    vintage = ['syrena', 'nysa', 'warszawa', 'polonez', 'żuk', 'gaz', 'moskwicz',
               'lada', 'wartburg', 'trabant', 'tata']
    premium_asian = ['infiniti', 'acura', 'baic', 'ssangyong']
    budget = ['dacia', 'fiat', 'daewoo', 'lancia']

    if brand_lower in luxury: return 'Luxury'
    if brand_lower in american: return 'American'
    if brand_lower in vintage: return 'Vintage'
    if brand_lower in premium_asian: return 'Premium_Asian'
    if brand_lower in budget: return 'Budget'
    return 'Standard'

def generate_otomoto_link(brand, model, year, price_min, price_max):
    base_url = "https://www.otomoto.pl/osobowe"
    brand_clean = brand.lower().replace(' ', '-').replace('ö', 'o').replace('é', 'e')
    model_clean = model.lower().replace(' ', '-') if model else ""
    params = {
        'search[filter_enum_make]': brand,
        'search[filter_float_year:from]': max(year - 2, 1980),
        'search[filter_float_year:to]': min(year + 2, 2026),
        'search[filter_float_price:from]': int(price_min * 0.89),
        'search[filter_float_price:to]': int(price_max * 1.10),
    }
    if model_clean:
        return f"{base_url}/{brand_clean}/{model_clean}?{urlencode(params)}"
    return f"{base_url}/{brand_clean}?{urlencode(params)}"

def prepare_input_data(user_inputs):
    df = pd.DataFrame([user_inputs])
    current_year = 2026
    df['Vehicle_age'] = current_year - df['Production_year']

    if isinstance(df['Features'].iloc[0], str):
        feat_list = [f.strip() for f in df['Features'].iloc[0].split(',') if f.strip()]
        df['Features'] = [feat_list]

    df['Num_features'] = df['Features'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['Age_category'] = pd.cut(df['Vehicle_age'], bins=[-1, 3, 10, 20, 100],
                                labels=['New', 'Standard', 'Old', 'Vintage'])
    df['Is_new_car'] = (df['Vehicle_age'] <= 1).astype(int)
    df['Is_old_car'] = (df['Vehicle_age'] > 20).astype(int)
    df['Mileage_per_year'] = df['Mileage_km'] / (df['Vehicle_age'] + 1)
    df['Usage_intensity'] = df['Mileage_km'] / (df['Vehicle_age'] + 1)
    df['HP_per_liter'] = df['Power_HP'] / (df['Displacement_cm3'] / 1000 + 0.1)
    df['Performance_category'] = pd.cut(df['Power_HP'], bins=[-1, 100, 200, 400, 2000],
                                        labels=['Economy', 'Standard', 'Sport', 'Supercar'])
    df['Brand_category'] = get_brand_reliability_category(df['Vehicle_brand'].iloc[0])
    df['Is_premium'] = df['Brand_category'].isin(['Luxury', 'Premium_Asian']).astype(int)
    df['Is_supercar'] = ((df['Power_HP'] > 500) | (df['Brand_category'] == 'Luxury')).astype(int)
    df['is_collector'] = ((df['Vehicle_age'] > 30) | (df['Brand_category'] == 'Luxury')).astype(int)
    df['Mileage_km_log'] = np.log1p(df['Mileage_km'])
    df['Power_HP_log'] = np.log1p(df['Power_HP'])
    df['Displacement_cm3_log'] = np.log1p(df['Displacement_cm3'])
    df['Vehicle_age_squared'] = df['Vehicle_age'] ** 2
    df['Power_HP_squared'] = df['Power_HP'] ** 2
    df['Mileage_km_squared'] = df['Mileage_km'] ** 2
    df['Age_Mileage_interaction'] = df['Vehicle_age'] * df['Mileage_km']
    df['Power_Age_interaction'] = df['Power_HP'] * df['Vehicle_age']
    df['Mileage_per_year_Age'] = df['Mileage_per_year'] * df['Vehicle_age']
    df['Brand_frequency'] = 100
    df['Brand_popularity'] = 'Common'

    expected_columns = [
        'Condition', 'Vehicle_brand', 'Vehicle_model', 'Mileage_km', 'Power_HP',
        'Displacement_cm3', 'Fuel_type', 'Drive', 'Transmission', 'Type',
        'Doors_number', 'Colour', 'Origin_country', 'First_owner',
        'Offer_location', 'Features', 'Vehicle_age', 'Num_features',
        'Age_category', 'Is_new_car', 'Is_old_car', 'Mileage_per_year',
        'Usage_intensity', 'HP_per_liter', 'Performance_category', 'Is_premium',
        'Is_supercar', 'is_collector', 'Mileage_km_log', 'Power_HP_log',
        'Displacement_cm3_log', 'Vehicle_age_squared', 'Power_HP_squared',
        'Mileage_km_squared', 'Age_Mileage_interaction',
        'Power_Age_interaction', 'Mileage_per_year_Age', 'Brand_category',
        'Brand_frequency', 'Brand_popularity'
    ]
    return df[expected_columns]

model_data = load_model()
if model_data:
    pipeline = model_data['model_pipeline'] if isinstance(model_data, dict) else model_data
else:
    st.error("⚠️ Model cannot be loaded, check your connection.")
    st.stop()

if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to(page):
    st.session_state.page = page
    st.rerun()

with st.sidebar:
    st.markdown("""
    <div style='padding: 24px 8px 20px; text-align: center;'>
        <div style='font-family: Syne, sans-serif; font-size: 22px; font-weight: 800; color: #f7fafc; letter-spacing: -0.02em;'>
            🚗 CarVal PL
        </div>
        <div style='font-size: 12px; color: #4a5568; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.1em;'>
            Poland Market 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    pages = [
        ("🏠", "Home", "home"),
        ("🔮", "Price Prediction", "predict"),
        ("🗺️", "Regional Market", "regional"),
        ("📊", "Visualizations", "visualizations"),
        ("🧠", "About Model", "info"),
    ]

    for icon, label, key in pages:
        btn_type = "primary" if st.session_state.page == key else "secondary"
        if st.button(f"{icon}  {label}", use_container_width=True, type=btn_type, key=f"nav_{key}"):
            navigate_to(key)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.06); margin: 24px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 0 8px;'>
        <p style='font-size: 11px; color: #2d3748; line-height: 1.8; margin: 0;'>
            XGBoost · scikit-learn<br>200,000+ listings · R² 92.4%
        </p>
    </div>
    """, unsafe_allow_html=True)

def home_page():
    st.markdown("""
    <div style='padding: 60px 0 40px; text-align: center;'>
        <div class='hero-badge'>🇵🇱 Polish Automotive Market · 2026</div>
        <h1 style='font-size: 64px; font-weight: 800; color: #f7fafc !important;
                   letter-spacing: -0.04em; line-height: 1.05; margin: 0 0 20px;'>
            What is your car<br>worth?
        </h1>
        <p style='font-size: 20px; color: #718096; max-width: 520px;
                  margin: 0 auto; line-height: 1.6;'>
            Artificial intelligence will value your car based on
            200,000+ listings from the Polish market.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_cta, col_cta2, col_cta3 = st.columns([1, 1.5, 1])
    with col_cta2:
        if st.button("🔮  Get your car valuation now", use_container_width=True):
            navigate_to('predict')

    st.markdown("<br><hr>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    features_data = [
        ("🔮", "Instant Valuation", "Enter your car's details and get an instant market value.", "predict", "Value my car"),
        ("🗺️", "Regional Analysis", "Explore how prices differ between cities across Poland.", "regional", "View map"),
        ("📊", "Data & Charts", "Interactive visualizations from the analysis of 200,000+ listings.", "visualizations", "Browse charts"),
    ]
    for col, (icon, title, desc, target, btn_label) in zip([c1, c2, c3], features_data):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align: center;'>
                <span class='feature-icon'>{icon}</span>
                <div class='feature-title'>{title}</div>
                <div class='feature-desc'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            if st.button(btn_label, key=f"card_btn_{target}", use_container_width=True):
                navigate_to(target)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='text-align: center; margin-bottom: 32px;'>Model Performance</h2>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="max-width:900px;margin:auto;text-align:center;font-size:18px;line-height:1.6;">

    The initial XGBoost model improved prediction performance compared to the previously 
    tested models, slightly increasing all evaluation metrics. Most predicted values are 
    close to the actual car prices, indicating strong overall accuracy. However, the model 
    still produced several extreme outliers, with errors reaching up to 1.5 million PLN.

    Further analysis showed that these large errors mainly occur for very old vehicles 
    (before 1980) and modern luxury or supercars. These segments are difficult to 
    because they are rare in the dataset, which limits the model’s ability to learn 
    pricing patterns.

    After applying hyperparameter tuning and additional feature engineering, the tuned 
    XGBoost model did not significantly improve the metrics. The R² score decreased by 
    about 2%, while MAPE increased by around 0.5%, and RMSE and MAE rose slightly. 
    Despite this, the tuned model appears to produce more stable predictions due to the 
    newly engineered features.

    Analysis of MAPE by brand showed that the impact of feature engineering varied across 
    manufacturers. Prediction errors decreased for some brands but increased slightly for 
    others, suggesting that the new features captured certain brand-specific 
    pricing relationships.

    Overall, both XGBoost models demonstrate strong predictive performance, with an average 
    absolute error of around 7,900–8,100 PLN. Considering that car prices in the dataset 
    range from tens of thousands to several million PLN, this level of error remains relatively small.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><hr>", unsafe_allow_html=True)

    st.markdown("""
    <h2 style='text-align: center; margin-bottom: 24px;'>About the Project</h2>

    <div style="max-width:900px;margin:auto;text-align:center;font-size:18px;line-height:1.6;">

    <p>
    <b style='color: #f7fafc;'>Car Price Prediction</b> is a machine learning project designed to estimate 
    used car prices on the Polish automotive market. The model analyzes historical listing data and 
    predicts vehicle values with an accuracy of approximately 
    <b style='color: #63b3ed;'>R² = 92.4%</b>.
    </p>

    <p>
    The final solution is based on the <b style='color: #f7fafc;'>XGBoost regression model</b>, which 
    outperformed other tested approaches such as Linear Regression and Random Forest. 
    The model was trained on a dataset containing more than <b style='color: #f7fafc;'>200,000 car listings</b> 
    and utilizes <b style='color: #f7fafc;'>over 30 engineered features</b> derived from vehicle specifications, 
    market attributes, and statistical feature interactions.
    </p>

    <p>
    Feature engineering incorporated both domain knowledge and data-driven transformations, 
    allowing the model to better capture relationships between factors such as vehicle age, mileage, 
    engine performance, and brand reputation. These variables are among the most influential drivers 
    of vehicle price in the used car market.
    </p>

    <p>
    The model achieves strong predictive performance across most market segments, with relatively 
    low average prediction error considering the wide price range present in the dataset.
    </p>

    <div class='warning-box' style='margin-top: 20px;'>
    <p>
    ⚠️ <b>Important:</b> The model performs best for mass-market vehicles (economy, standard, and premium segments). 
    Prediction accuracy may be lower for <b>luxury vehicles, rare models, vintage cars, and supercars</b>, 
    as these segments are less represented in the training data and often follow different pricing dynamics.
    </p>
    </div>

    </div>
    """, unsafe_allow_html=True)

def predict_page():
    st.markdown("""
    <h1 style='margin-bottom: 8px;'>🔮 Vehicle Valuation</h1>
    <p style='color: #718096; margin-bottom: 32px; font-size: 16px;'>
        Fill in the form — fields marked with * are required.
    </p>
    """, unsafe_allow_html=True)

    if st.button("← Back"): navigate_to("home")

    brand_list = sorted([
        'Abarth', 'Acura', 'Aixam', 'Alfa Romeo', 'Alpine', 'Aston Martin',
        'Audi', 'Austin', 'Autobianchi', 'Baic', 'Bentley', 'BMW', 'Buick',
        'Cadillac', 'Chevrolet', 'Chrysler', 'Citroën', 'Cupra', 'Dacia',
        'Daewoo', 'Daihatsu', 'DFSK', 'DKW', 'Dodge', 'DS Automobiles',
        'FAW', 'Ferrari', 'Fiat', 'Ford', 'Gaz', 'GMC', 'Honda', 'Hummer',
        'Hyundai', 'Infiniti', 'Isuzu', 'Iveco', 'Jaguar', 'Jeep', 'Kia',
        'Lada', 'Lamborghini', 'Lancia', 'Land Rover', 'Lexus', 'Lincoln',
        'Lotus', 'Warszawa', 'Maserati', 'Maybach', 'Mazda', 'McLaren',
        'Mercedes-Benz', 'Mercury', 'MG', 'Microcar', 'MINI', 'Mitsubishi',
        'Moskwicz', 'Nissan', 'NSU', 'Nysa', 'Oldsmobile', 'Opel',
        'Toyota', 'Tata', 'Uaz', 'Żuk', 'Trabant', 'Suzuki', 'Inny',
        'Volvo', 'Subaru', 'Volkswagen', 'SsangYong', 'Saab', 'Plymouth',
        'Renault', 'Peugeot', 'Rolls-Royce', 'RAM', 'Triumph', 'Rover',
        'Wołga', 'Tarpan', 'Polonez', 'Pontiac', 'Porsche', 'Santana',
        'Saturn', 'Scion', 'Seat', 'Škoda', 'Smart', 'Syrena', 'Talbot',
        'Tavria', 'Tesla', 'Vanderhall', 'Vauxhall', 'Wartburg',
        'Zaporożec', 'Zastava'
    ])

    country_list = [
        "Germany", "USA", "Japan", "France", "Italy", "United Kingdom",
        "South Korea", "China", "Sweden", "Spain", "Netherlands", "Belgium",
        "Canada", "Australia", "India", "Russia", "Mexico", "Brazil",
        "Czech Republic", "Poland", "Turkey", "Austria", "Switzerland",
        "Denmark", "Norway", "Finland", "Portugal", "Greece", "Thailand", "Vietnam"
    ]

    fuel_list = ['Gasoline', 'Gasoline + LPG', 'Diesel', 'Hybrid',
                 'Gasoline + CNG', 'Hydrogen', 'Electric']
    drive_list = ['Front wheels', 'Rear wheels', '4x4 (attached automatically)',
                  '4x4 (permanent)', '4x4 (attached manually)']
    body_list = ['small_cars', 'coupe', 'city_cars', 'convertible', 'compact',
                 'SUV', 'sedan', 'station_wagon', 'minivan']
    color_list = ["Black", "White", "Grey", "Silver", "Blue", "Red",
                  "Yellow", "Green", "Other"]

    with st.form("prediction_form"):
        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### 🚗 Vehicle Details")
            brand_choice = st.selectbox("Brand *", ["— select —"] + brand_list)
            vehicle_model = st.text_input("Model", placeholder="e.g. Golf, Astra, Corolla…")
            production_year = st.slider("Production Year *", 1950, 2026, 2015)
            mileage_km = st.slider("Mileage (km) *", 0, 1_000_000, 100_000, step=500)
            power_HP = st.slider("Power (HP) *", 10, 1_500, 150)
            displacement_cm3 = st.slider("Displacement (cm³) *", 500, 8_000, 2_000, step=1)

        with col2:
            st.markdown("#### ⚙️ Specifications")
            fuel_type = st.selectbox("Fuel Type *", ["— select —"] + fuel_list)
            transmission = st.selectbox("Transmission *", ["— select —", "Manual", "Automatic"])
            drive = st.selectbox("Drive *", ["— select —"] + drive_list)
            car_type = st.selectbox("Body Type *", ["— select —"] + body_list)
            colour = st.selectbox("Colour", ["— select —"] + color_list)
            condition = st.selectbox("Condition *", ["— select —", "Used", "New"])

        st.markdown("#### 📍 Additional Information")
        a1, a2, a3 = st.columns(3)
        with a1:
            origin_country = st.selectbox("Country of Origin", ["— select —"] + country_list)
        with a2:
            first_owner = st.selectbox("First Owner", ["— select —", "Yes", "No"])
        with a3:
            offer_location = st.text_input("Location", placeholder="e.g. Warszawa, Kraków, ...")

        features = st.text_input(
            "Features (comma separated)",
            placeholder="e.g. ABS, GPS, Leather seats, Climatronic"
        )

        submitted = st.form_submit_button("🔮  Calculate Valuation", use_container_width=True)

    if submitted:
        errors = []
        if brand_choice == "— select —":
            errors.append("Brand")
        if fuel_type == "— select —":
            errors.append("Fuel Type")
        if transmission == "— select —":
            errors.append("Transmission")
        if drive == "— select —":
            errors.append("Drive")
        if car_type == "— select —":
            errors.append("Body Type")
        if condition == "— select —":
            errors.append("Condition")

        if errors:
            st.error(f"⚠️ Please fill in the required fields: **{', '.join(errors)}**")
        else:
            try:
                user_inputs = {
                    "Condition": condition,
                    "Vehicle_brand": brand_choice,
                    "Vehicle_model": vehicle_model or "Unknown",
                    "Production_year": production_year,
                    "Mileage_km": mileage_km,
                    "Power_HP": power_HP,
                    "Displacement_cm3": displacement_cm3,
                    "Fuel_type": fuel_type,
                    "Drive": drive,
                    "Transmission": transmission,
                    "Type": car_type,
                    "Doors_number": 5,
                    "Colour": colour,
                    "Origin_country": origin_country,
                    "First_owner": "No" if first_owner == "— select —" else first_owner,
                    "Offer_location": offer_location or "Unknown",
                    "Features": features or ""
                }

                final_df = prepare_input_data(user_inputs)
                y_log = pipeline.predict(final_df)[0]
                price = np.expm1(y_log)
                price_min = price * 0.85
                price_max = price * 1.15

                st.markdown(f"""
                <div class='price-card'>
                    <div class='price-label'>💰 Estimated Market Value</div>
                    <div class='price-value'>{price:,.0f} PLN</div>
                    <div class='price-range'>Range: {price_min:,.0f} – {price_max:,.0f} PLN</div>
                </div>
                """, unsafe_allow_html=True)

                d1, d2, d3 = st.columns(3)
                vehicle_age = 2026 - production_year
                mileage_per_year = mileage_km / max(vehicle_age, 1)
                hp_per_liter = power_HP / (displacement_cm3 / 1000)

                with d1:
                    st.metric("Vehicle Age", f"{vehicle_age} years")
                with d2:
                    st.metric("Avg. Annual Mileage", f"{mileage_per_year:,.0f} km")
                with d3:
                    st.metric("Specific Power", f"{hp_per_liter:.1f} HP/L")

                otomoto_url = generate_otomoto_link(
                    brand_choice, vehicle_model or "", production_year, price_min, price_max
                )
                st.markdown(f"""
                <div class='info-box' style='margin-top: 24px;'>
                    <p style='margin: 0 0 4px; font-size: 15px;'>
                        🔍 Looking for a <b>{brand_choice} {vehicle_model or ''}</b>
                        around <b>{price:,.0f} PLN</b>?
                    </p>
                    <a class='otomoto-btn' href='{otomoto_url}' target='_blank'>
                        🚗  Browse similar cars on Otomoto
                    </a>
                </div>
                """, unsafe_allow_html=True)

                brand_cat = get_brand_reliability_category(brand_choice)
                if brand_cat in ['Luxury', 'Vintage'] or power_HP > 500:
                    st.markdown("""
                    <div class='warning-box' style='margin-top: 16px;'>
                        <p>⚠️ <b>Note:</b> This vehicle belongs to a special category
                        (luxury / vintage / high-performance). Valuation accuracy may be
                        lower due to limited market data.</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Valuation error: {str(e)}")
                st.info("Please check your inputs and try again.")

def regional_page():
    st.markdown("""
    <h1 style='margin-bottom: 8px;'>🗺️ Regional Market</h1>
    <p style='color: #718096; margin-bottom: 24px; font-size: 16px;'>
        Distribution of car listings across Poland based on 200,000+ entries.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <p>📌 Illustrative data — geographic distribution of listings from Polish automotive platforms.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("← Back"): navigate_to("home")

    city_sales, city_coords = load_sample_data()

    poland_map = folium.Map(location=[52.0, 19.0], zoom_start=6, tiles='CartoDB dark_matter')
    max_sales = max(city_sales.values())

    for city, sales in city_sales.items():
        if city in city_coords:
            lat, lon = city_coords[city]
            radius = 10 + (sales / max_sales) * 30
            color = '#e53e3e' if sales > 3000 else '#ed8936' if sales > 2000 else '#ecc94b' if sales > 1000 else '#48bb78'
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                popup=f"<b>{city}</b><br>{sales:,} listings",
                color=color, fill=True, fillColor=color,
                fillOpacity=0.75, weight=2
            ).add_to(poland_map)

    st_folium(poland_map, width=None, height=560)

    st.markdown("### 📊 Top 10 Cities")
    sorted_cities = sorted(city_sales.items(), key=lambda x: x[1], reverse=True)[:10]
    chart_data = pd.DataFrame(sorted_cities, columns=['City', 'Listings'])

    fig = px.bar(
        chart_data, x='City', y='Listings',
        color='Listings', color_continuous_scale='Blues',
        title='Number of Listings by City'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e0', title_font_size=18,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div class='feature-title'>🏙️ Urban vs Rural</div>
            <div class='feature-desc'>
                Major cities (Gdańsk, Kraków, Wrocław) account for over 60% of listings.
                Higher vehicle turnover in metropolitan areas.
                Premium brands are more common in large cities.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div class='feature-title'>📈 Market Trends</div>
            <div class='feature-desc'>
                Western Poland shows higher average prices.
                Coastal cities prefer imported German vehicles.
                Eastern regions more frequently choose budget brands.
                Southern Poland dominates in number of offers.
            </div>
        </div>
        """, unsafe_allow_html=True)

def visualizations_page():
    st.markdown("""
    <h1 style='margin-bottom: 8px;'>📊 Data Visualizations</h1>
    <p style='color: #718096; margin-bottom: 32px; font-size: 16px;'>
        Key findings from the analysis of 200,000+ car listings.
    </p>
    """, unsafe_allow_html=True)

    if st.button("← Back"): navigate_to("home")

    np.random.seed(42)

    PLOT_LAYOUT = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e0',
        title_font_size=18,
        title_font_family='Syne',
        margin=dict(t=48, b=32, l=16, r=16)
    )

    st.markdown("### 💰 Price Distribution")
    st.image(
        "images/eda_price_distribution.png",
        use_container_width=True
    ) 
    st.markdown("""
    <div class='info-box'><p>
    📌 The left chart shows a strongly right-skewed price distribution. 
    Most vehicles are priced below 100,000 PLN, with a median of around 35,500 PLN 
    The mean is much higher than the median indicating that a small number of expensive 
    premium vehicles pull the average upward. In this case, the median better represents 
    the typical car price

    The right chart shows the distribution after a logarithmic transformation which 
    reduces skewness and makes the data more symmetrical and closer to a normal distribution 
    Using a log-transformed target variable can improve regression performance by stabilizing 
    variance, reducing the impact of extreme outliers, and better capturing proportional 
    relationships between features and price.
    </p></div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 📉 Value Depreciation")
    st.image(
        "images/eda_depreciation_analysis.png",
        use_container_width=True
    )
    st.markdown("""
    <div class='info-box'><p>
        📌 Vehicle depreciation is most pronounced during the first three years, when a 
            typical car loses 40–50% of its initial value. This period reflects the 
            steepest decline in market price, as new vehicles quickly lose their 
            “new car” premium. After this initial phase, the rate of depreciation 
            slows, and prices begin to gradually stabilize, especially once vehicles 
            reach around 10–15 years of age.
            Interestingly, some vehicles that are older than 25 years may experience an 
            increase in value, as certain models transition into the vintage or collector
            category. These cars become highly sought after by enthusiasts, collectors,
            and niche markets, often commanding prices that can exceed their original market
            value.
                
    </p></div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 🏆 Average Price by Brand")
    st.image(
        "images/eda_median_preice_top20_brands.png",
        use_container_width=True
    )

    st.markdown("""
    <div class='info-box'><p>
        📌 The bar chart shows the most popular vehicle brands along with their median 
            prices. Brands with prices above the overall median are highlighted, 
            representing the premium segment of the market.
            At the top of the ranking is Mercedes-Benz, with a median price of approximately
             62,000 PLN. The second and third spots are also taken by German brands — BMW 
            (~60,000 PLN) and Audi (~50,000 PLN). Another premium European brand, Volvo, 
            also maintains a relatively high median price of nearly 49,000 PLN.
            The remaining brands have noticeably lower median prices, typically ranging 
            from 20,000 PLN to 40,000 PLN, and include vehicles from Asian, French, and 
            American manufacturers. Their median prices are roughly 2–3 times lower than 
            those of the leading German brands, highlighting a clear segmentation in the market.
                            
    </p></div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 🎯 Most Important Model Features")
    st.image(
        "images/SHAP_feature_importance.png",
        use_container_width=True
    )

    st.markdown("""
    <div class='info-box'><p>
        📌 The most influential feature in the model is Vehicle Age, with a SHAP importance 
            of around 0.55. The second most important factor is Engine Power (HP), roughly 
            three times less influential. Other significant predictors include Vehicle Model 
            (~0.15) and Mileage (km) (~0.12).
            These results align well with real-world car pricing: age, engine power, and 
            mileage are the primary drivers of value. In the final XGBoost model, additional
            emphasis was placed on Vehicle Age to better differentiate typical cars from 
            niche or collector vehicles, such as supercars. This confirms that the model’s 
            feature importance is both reasonable and interpretable.
            At the lower end, Vehicle Type and Fuel Type have minimal impact, with SHAP 
            values between 0.01 and 0.03. While these features contribute slightly, they 
            play a much smaller role compared to the main mechanical and usage-related 
            characteristics.
        
    </p></div>
    """, unsafe_allow_html=True)

def info_page():
    st.markdown("""
    <h1 style='margin-bottom: 8px;'>🧠 How the Model Works</h1>
    <p style='color: #718096; margin-bottom: 32px; font-size: 16px;'>
        Architecture, training process and limitations of the XGBoost model.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("""
### 🎯 Model Overview
The application uses an **XGBoost** model trained on data from the Polish automotive market
to estimate **used car prices**.

---

### 🚀 Why XGBoost?

| Model | R² | MAPE | Status |
|-------|-----|------|--------|
| Linear Regression | 0.831 | 29.3% | ❌ Baseline |
| Random Forest | 0.938 | 20.0% | ✅ Good |
| **XGBoost (tuned)** | **0.924** | **17.2%** | ⭐ Selected |

**Key advantages:**
- 🎯 Gradient Boosting — sequentially improves predictions
- 🛡️ Robust to noise and missing values
- ⚡ Lightning-fast predictions (< 1 s)
- 🔧 L1/L2 regularisation — prevents overfitting

---

### 📊 Model Features

**Base features:** Brand, Model, Year, Mileage, Power, Displacement, Fuel, Transmission, Drive, Body type, Colour, Location, Condition, Country of origin

**Engineered features:**
- `HP_per_liter`, `Mileage_per_year`, `Usage_intensity`
- `Age_category`, `Brand_category`, `Is_premium`, `is_collector`
- Interactions: `Age × Mileage`, `Power × Age`
- Log-transforms and squared features

---

### 🎯 Model Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R²** | 92.4% | Explains ~92.5% of price variance |
| **RMSE** | 23,048 PLN | Typical absolute error |
| **MAE** | 8,109 PLN | Mean absolute error |
| **MAPE** | 17.2% | ~17% relative deviation |

---

### 🔬 Training Process

1. **Data collection** — 200,000+ listings from Polish platforms
2. **Preprocessing** — duplicate removal, outlier handling, normalisation
3. **Feature Engineering** — support for the model
4. **Model selection** — comparison of 3 algorithms
5. **Tuning** — Optuna (50+ Bayesian optimisation trials)
6. **Validation** — 80/20 split, cross-validation
7. **Deployment** — Streamlit + Hugging Face Hub

---

### ⚠️ Limitations

✅ **Model performs well for:**
- Mass-market cars (VW, Toyota, Ford, Opel, Škoda)
- Standard configurations (100–300 HP, 50–200k km)
- Vehicles from 2010–2024

⚠️ **Lower accuracy for:**
- Luxury brands (Ferrari, Lamborghini, Rolls-Royce)
- Vintage cars (> 30 years old)
- Rare models (< 20 entries in the dataset)
- Custom factory modifications not captured in features

---

### 🛠️ Tech Stack

- **ML:** XGBoost, scikit-learn, category-encoders
- **Optimisation:** Optuna (Bayesian search)
- **Data processing:** Pandas, NumPy
- **Deployment:** Streamlit, Hugging Face Hub
- **Visualisations:** Plotly, Folium
""")

page = st.session_state.page
if page == 'home':
    home_page()
elif page == 'predict':
    predict_page()
elif page == 'regional':
    regional_page()
elif page == 'visualizations':
    visualizations_page()
elif page == 'info':
    info_page()