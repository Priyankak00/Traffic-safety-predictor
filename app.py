import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import time

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="SafeStreets AI | Collision Risk Inspector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .safe-metric { border-left-color: #00cc66 !important; }
    h1 { color: #2c3e50; }
    .stProgress > div > div > div > div { background-image: linear-gradient(to right, #00cc66, #ffcc00, #ff4b4b); }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INTELLIGENT MODEL TRAINING (CACHED)
# ==========================================
@st.cache_resource
def load_and_train_brain():
    # --- A. Load & Clean ---
    file_path = 'balanced_vehicle_collisions_sample.csv'
    try:
        df = pd.read_csv(file_path)
    except:
        st.error("🚨 CRITICAL: CSV file not found. Please upload 'balanced_vehicle_collisions_sample.csv'.")
        return None

    # Parse Dates
    df['CRASH_DT'] = pd.to_datetime(df['CRASH DATE'] + ' ' + df['CRASH TIME'], errors='coerce')
    df = df.dropna(subset=['CRASH_DT'])

    # Smart Imputation (Don't drop, fill!)
    df['BOROUGH'] = df['BOROUGH'].fillna('Unknown')
    df['VEHICLE TYPE CODE 1'] = df['VEHICLE TYPE CODE 1'].fillna('Unspecified')
    df['CONTRIBUTING FACTOR VEHICLE 1'] = df['CONTRIBUTING FACTOR VEHICLE 1'].fillna('Unspecified')

    # --- B. Feature Engineering ---
    df['Hour'] = df['CRASH_DT'].dt.hour
    df['DayOfWeek'] = df['CRASH_DT'].dt.dayofweek
    df['Month'] = df['CRASH_DT'].dt.month
    df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
    df['Rush_Hour'] = df['Hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)

    # Group Rare Categories (Noise Reduction)
    top_vehicles = df['VEHICLE TYPE CODE 1'].value_counts().nlargest(15).index
    df['VEHICLE_GROUP'] = df['VEHICLE TYPE CODE 1'].apply(lambda x: x if x in top_vehicles else 'Other')

    top_factors = df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().nlargest(10).index
    df['FACTOR_GROUP'] = df['CONTRIBUTING FACTOR VEHICLE 1'].apply(lambda x: x if x in top_factors else 'Other')

    # --- C. Encoding ---
    le_borough = LabelEncoder()
    df['BOROUGH_Code'] = le_borough.fit_transform(df['BOROUGH'])

    le_vehicle = LabelEncoder()
    df['VEHICLE_Code'] = le_vehicle.fit_transform(df['VEHICLE_GROUP'])

    le_factor = LabelEncoder()
    df['FACTOR_Code'] = le_factor.fit_transform(df['FACTOR_GROUP'])

    # Define Feature List (Order Matters!)
    features = ['Hour', 'DayOfWeek', 'Month', 'Is_Weekend', 'Rush_Hour', 'BOROUGH_Code', 'VEHICLE_Code']
    X = df[features]

    # --- D. Training Models ---
    
    # 1. Severity (Classification)
    df['IS_SEVERE'] = ((df['NUMBER OF PERSONS INJURED'] > 0) | (df['NUMBER OF PERSONS KILLED'] > 0)).astype(int)
    clf_severity = RandomForestClassifier(n_estimators=60, max_depth=12, class_weight='balanced', random_state=42)
    clf_severity.fit(X, df['IS_SEVERE'])

    # 2. Injury Count (Regression)
    y_injuries = df['NUMBER OF PERSONS INJURED'].fillna(0)
    reg_injured = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    reg_injured.fit(X, y_injuries)

    # 3. Cause (Classification)
    clf_cause = RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42)
    clf_cause.fit(X, df['FACTOR_Code'])

    # 4. Hotspots (Clustering)
    # Filter valid coordinates for mapping
    df_geo = df.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()
    df_geo = df_geo[(df_geo['LATITUDE'] != 0) & (df_geo['LONGITUDE'] != 0)]
    kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
    kmeans.fit(df_geo[['LATITUDE', 'LONGITUDE']])
    df_geo['Cluster_Label'] = kmeans.labels_ # FIX: Save labels to DF for plotting

    return {
        'clf_sev': clf_severity, 'reg_inj': reg_injured, 'clf_cause': clf_cause, 'kmeans': kmeans,
        'le_b': le_borough, 'le_v': le_vehicle, 'le_f': le_factor,
        'features': features, 'df_geo': df_geo, 'raw_df': df
    }

# Load Models
with st.spinner("⚙️ Booting up AI Engines... (Analyzing historical crash data)"):
    brain = load_and_train_brain()

if not brain:
    st.stop()

# ==========================================
# 3. SIDEBAR: SCENARIO BUILDER
# ==========================================
st.sidebar.title("🛠️ Scenario Builder")
st.sidebar.markdown("Define the crash conditions:")

# Inputs
s_date = st.sidebar.date_input("Date", value=pd.to_datetime("today"))
s_time = st.sidebar.time_input("Time", value=pd.to_datetime("now").time())
s_borough = st.sidebar.selectbox("Borough", options=brain['le_b'].classes_)
s_vehicle = st.sidebar.selectbox("Vehicle Type", options=brain['le_v'].classes_)

# Dynamic Location Defaults
# Find the center of the selected borough to update lat/long defaults
borough_center = brain['df_geo'][brain['raw_df']['BOROUGH'] == s_borough][['LATITUDE', 'LONGITUDE']].mean()
if pd.isna(borough_center['LATITUDE']):
    # Fallback if borough has no geo data
    borough_center = pd.Series({'LATITUDE': 40.7128, 'LONGITUDE': -74.0060})

st.sidebar.markdown("---")
st.sidebar.markdown("📍 **Location Coordinates**")
s_lat = st.sidebar.number_input("Latitude", value=float(borough_center['LATITUDE']), format="%.5f")
s_long = st.sidebar.number_input("Longitude", value=float(borough_center['LONGITUDE']), format="%.5f")

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
st.title("🛡️ SafeStreets AI")
st.markdown(f"**Predictive Analysis for: {s_borough} on {s_date} at {s_time}**")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["🚀 Risk Prediction", "🗺️ Geospatial Intel", "📊 Model Insights"])

# --- TAB 1: PREDICTION DASHBOARD ---
with tab1:
    if st.button("Analyze Risk Scenario", type="primary", use_container_width=True):
        
        # 1. Prepare Input Data (Robustly match training columns)
        dt = pd.to_datetime(f"{s_date} {s_time}")
        input_data = {
            'Hour': [dt.hour],
            'DayOfWeek': [dt.dayofweek],
            'Month': [dt.month],
            'Is_Weekend': [1 if dt.dayofweek >= 5 else 0],
            'Rush_Hour': [1 if (7 <= dt.hour <= 9) or (16 <= dt.hour <= 19) else 0],
            'BOROUGH_Code': [brain['le_b'].transform([s_borough])[0]],
            'VEHICLE_Code': [brain['le_v'].transform([s_vehicle])[0]]
        }
        X_new = pd.DataFrame(input_data) # Columns automatically match dict keys, which match training order

        # 2. Make Predictions
        sev_prob = brain['clf_sev'].predict_proba(X_new)[0][1] # Prob of Severe
        injuries = brain['reg_inj'].predict(X_new)[0]
        cause_code = brain['clf_cause'].predict(X_new)[0]
        likely_cause = brain['le_f'].inverse_transform([cause_code])[0]
        cluster_id = brain['kmeans'].predict([[s_lat, s_long]])[0]

        # 3. Display Results
        
        # Top Row: Safety Score & Cause
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Safety Assessment")
            # Safety Score is the inverse of Severity Probability
            safety_score = int((1 - sev_prob) * 100)
            
            # Progress Bar Color Logic
            bar_color = "red" if safety_score < 50 else "orange" if safety_score < 80 else "green"
            
            st.markdown(f"**Safety Score: {safety_score}/100**")
            st.progress(safety_score, text=f"Risk Probability: {sev_prob:.1%}")
            
            if sev_prob > 0.5:
                st.error(f"⚠️ HIGH RISK ALERT: This scenario has a {sev_prob:.1%} chance of resulting in injury.")
            else:
                st.success(f"✅ LOW RISK: Accidents in these conditions are usually minor.")

        with c2:
            st.info(f"**Most Likely Cause**\n\n### {likely_cause}")
            st.caption("Based on historical patterns for similar times/vehicles.")

        st.divider()

        # Bottom Row: Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Est. Injuries", f"{injuries:.2f}", help="Predicted number of people involved")
        m2.metric("Hotspot Zone", f"Cluster #{cluster_id}", help="Geospatial risk cluster ID")
        m3.metric("Rush Hour?", "Yes" if input_data['Rush_Hour'][0] else "No")

    else:
        st.info("👈 Adjust the scenario in the sidebar and click **'Analyze Risk Scenario'**.")

# --- TAB 2: GEOSPATIAL INTEL ---
with tab2:
    st.header("📍 Accident Hotspots & Location Context")
    
    # 1. Main Map (Clusters)
    # We use Plotly for a better interactive map than st.map
    fig_map = px.scatter_mapbox(
        brain['df_geo'], 
        lat="LATITUDE", 
        lon="LONGITUDE", 
        color="Cluster_Label",
        hover_name="BOROUGH",
        zoom=10, 
        height=500,
        title="Identified Accident Clusters (K-Means)"
    )
    # Add the USER'S current selected point
    fig_map.add_scattermapbox(
        lat=[s_lat], lon=[s_long], 
        mode='markers', marker=dict(size=15, color='red'), name='Selected Location'
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 3: MODEL INSIGHTS ---
with tab3:
    st.header("🧠 How the AI Decides")
    
    # Feature Importance Plot
    importance = brain['clf_sev'].feature_importances_
    feat_names = brain['features']
    
    # Create DataFrame for Plotly
    fi_df = pd.DataFrame({'Feature': feat_names, 'Importance': importance}).sort_values(by='Importance', ascending=True)
    
    fig_feat = px.bar(
        fi_df, x='Importance', y='Feature', orientation='h',
        title="Feature Importance (Severity Model)",
        color='Importance', color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_feat, use_container_width=True)
    
    st.markdown("""
    **Key:**
    * **Hour / Rush_Hour**: Time of day is often the biggest predictor.
    * **VEHICLE_Code**: Large vehicles often cause more severe accidents.
    * **Is_Weekend**: Weekend traffic patterns differ significantly from weekdays.
    """)

# Footer
st.markdown("---")
st.caption("© 2025 SafeStreets AI Project | Powered by Random Forest & Streamlit")