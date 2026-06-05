"""
RAPIDO ML PROJECT - STREAMLIT CLOUD VERSION
============================================
Deployed on Streamlit Cloud.
CSV uploaded via Git LFS — full EDA available.
Class mapping confirmed: 0=Cancelled, 1=Completed

Author: VishweshN
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Rapido ML Dashboard", page_icon="🛵",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; color: #ffffff; }
    [data-testid="stSidebar"] { background-color: #1a1a1a; border-right: 2px solid #ff6b00; }
    [data-testid="metric-container"] { background-color: #1e1e1e; border: 1px solid #ff6b00; border-radius: 10px; padding: 15px; }
    h1, h2, h3 { color: #ff6b00 !important; font-family: 'Arial Black', sans-serif; }
    .stButton > button { background-color: #ff6b00; color: white; border: none; border-radius: 8px; font-weight: bold; width: 100%; padding: 10px; }
    .stButton > button:hover { background-color: #e65c00; color: white; }
    .prediction-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 20px; font-weight: bold; margin: 10px 0; }
    .success-box { background-color: #1a3a1a; border: 2px solid #00cc44; color: #00cc44; }
    .warning-box { background-color: #3a2a1a; border: 2px solid #ff6b00; color: #ff6b00; }
    .danger-box  { background-color: #3a1a1a; border: 2px solid #ff4444; color: #ff4444; }
    .section-header { background: linear-gradient(90deg, #ff6b00, #1e1e1e); padding: 10px 20px; border-radius: 8px; margin: 10px 0; font-size: 18px; font-weight: bold; color: white; }
    .info-card { background-color: #1e1e1e; border-left: 4px solid #ff6b00; padding: 15px; border-radius: 5px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_PATH, 'models')
CSV_PATH   = os.path.join(BASE_PATH, 'master_dataset_engineered.csv')

# ── MODEL 1 DEFAULTS (retrained — no actual_ride_time_min, no incomplete_ride_reason) ──
MODEL1_DEFAULTS = {
    'day_of_week':2,'is_weekend':0,'hour_of_day':12,
    'estimated_ride_time_min':55,'ride_distance_km':13.03,
    'base_fare':187.35,'surge_multiplier':1.0,'booking_value':187.35,
    'cust_customer_age':41,'cust_customer_signup_days_ago':520,
    'cust_total_bookings':50,'cust_completed_rides':45,
    'cust_cancelled_rides':2,'cust_incomplete_rides':1,
    'cust_cancellation_rate':4.0,'cust_avg_customer_rating':4.5,
    'driver_driver_age':38,'driver_driver_experience_years':7,
    'driver_total_assigned_rides':21,'driver_accepted_rides':16,
    'driver_delay_count':0,'driver_delay_rate':0.02,
    'driver_acceptance_rate':0.90,'driver_cancellation_rate':0.02,
    'driver_total_cancellations':1,'driver_incomplete_rides':0,
    'total_risk_factors':0,'pairing_quality_score':75.0,
    'high_risk_pairing':0,'is_rush_hour':0,'surge_price_sensitive_cust':0,
    'city':'Delhi','pickup_location':'Loc_17','drop_location':'Loc_5',
    'vehicle_type':'Cab','traffic_level':'Low','weather_condition':'Clear',
    'cust_customer_gender':'Male','cust_customer_city':'Mumbai',
    'cust_preferred_vehicle_type':'Bike','cust_frequency_category':'Regular',
    'cust_value_segment':'High Value','cust_favorite_city':'Delhi',
    'driver_driver_city':'Mumbai','driver_vehicle_type':'Auto',
    'driver_experience_level':'Beginner','driver_performance_tier':'Gold',
    'driver_primary_city':'Mumbai','rush_hour_type':'Morning',
    'time_of_day':'Afternoon','hour_category':'Morning',
    'fare_category':'Medium','surge_category':'No Surge',
    'distance_category':'Medium','city_cleaned':'Delhi',
    'route':'Loc_17_to_Loc_5','cust_driver_exp_match':'Mismatched',
}

OTHER_DEFAULTS = {
    'day_of_week':2,'is_weekend':0,'hour_of_day':12,
    'estimated_ride_time_min':55,'actual_ride_time_min':58,
    'ride_distance_km':13.03,'base_fare':187.35,
    'surge_multiplier':1.6,'booking_value':290.09,
    'cust_customer_age':41,'cust_customer_signup_days_ago':520,
    'cust_total_bookings':11,'cust_completed_rides':7,
    'cust_cancelled_rides':2,'cust_incomplete_rides':1,
    'cust_cancellation_rate':22.22,'cust_avg_customer_rating':4.3,
    'cust_customer_cancel_flag':1,'driver_driver_age':38,
    'driver_driver_experience_years':7,'driver_total_assigned_rides':21,
    'driver_accepted_rides':16,'driver_delay_count':1,
    'driver_delay_rate':0.05,'driver_acceptance_rate':0.76,
    'driver_cancellation_rate':0.05,'driver_total_cancellations':2,
    'driver_incomplete_rides':1,'total_risk_factors':1,
    'pairing_quality_score':0.5,'high_risk_pairing':0,
    'incomplete_ride_reason':'Driver Delay',
    'city':'Delhi','pickup_location':'Loc_17','drop_location':'Loc_5',
    'vehicle_type':'Cab','traffic_level':'High','weather_condition':'Heavy Rain',
    'cust_customer_gender':'Male','cust_customer_city':'Mumbai',
    'cust_preferred_vehicle_type':'Bike','cust_frequency_category':'Occasional',
    'cust_value_segment':'Medium Value','cust_favorite_city':'Delhi',
    'driver_driver_city':'Mumbai','driver_vehicle_type':'Auto',
    'driver_experience_level':'Beginner','driver_performance_tier':'Silver',
    'driver_primary_city':'Mumbai','rush_hour_type':'Morning',
    'time_of_day':'Afternoon','hour_category':'Morning',
    'fare_category':'Medium','surge_category':'Low Surge',
    'distance_category':'Medium','city_cleaned':'Delhi',
    'route':'Loc_17_to_Loc_5','cust_driver_exp_match':'Mismatched',
}

def get_defaults(model_num):
    return MODEL1_DEFAULTS if model_num == 1 else OTHER_DEFAULTS

def encode_time_minutes(minutes, encoder):
    target = f'1970-01-01 00:00:00.000000{int(minutes):03d}'
    classes = list(encoder.classes_)
    if target in classes:
        return classes.index(target)
    try:
        min_ns = int(encoder.classes_[0].split('.')[-1])
        max_ns = int(encoder.classes_[-1].split('.')[-1])
        clamped = max(min_ns, min(max_ns, int(minutes)))
        tc = f'1970-01-01 00:00:00.000000{clamped:03d}'
        if tc in classes:
            return classes.index(tc)
    except Exception:
        pass
    return len(classes) // 2

@st.cache_data
def load_master_data():
    try:
        if os.path.exists(CSV_PATH):
            return pd.read_csv(CSV_PATH, nrows=50000)
        return None
    except Exception:
        return None

@st.cache_resource
def load_all_models():
    models = {}
    files = {
        1:{'model':'model1_ride_outcome_best.pkl','scaler':'model1_scaler.pkl','features':'model1_features.pkl','encoders':'model1_label_encoders.pkl'},
        2:{'model':'model2_fare_prediction_best.pkl','scaler':'model2_scaler.pkl','features':'model2_features.pkl','encoders':'model2_label_encoders.pkl'},
        3:{'model':'model3_customer_cancellation_best.pkl','scaler':'model3_scaler.pkl','features':'model3_features.pkl','encoders':'model3_label_encoders.pkl'},
        4:{'model':'model4_driver_delay_best.pkl','scaler':'model4_scaler.pkl','features':'model4_features.pkl','encoders':'model4_label_encoders.pkl'},
    }
    for num, comps in files.items():
        models[num] = {}
        for comp, fname in comps.items():
            try:
                with open(os.path.join(MODEL_PATH, fname), 'rb') as f:
                    models[num][comp] = pickle.load(f)
            except Exception:
                models[num][comp] = None
    return models

@st.cache_data
def load_comparison_csv(model_num):
    try:
        path = os.path.join(MODEL_PATH, f'model{model_num}_comparison.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            if df is not None and not df.empty:
                return df
        return None
    except Exception:
        return None

def prepare_input(user_inputs, model_num, models):
    try:
        features = models[model_num]['features']
        scaler   = models[model_num]['scaler']
        encoders = models[model_num]['encoders']
        if features is None:
            st.error("Feature list not loaded!")
            return None
        defaults = get_defaults(model_num)
        input_df = pd.DataFrame(index=[0], columns=features, dtype=object)
        for col in features:
            input_df[col] = defaults.get(col, 0)
        for key, val in user_inputs.items():
            if key in input_df.columns:
                input_df[key] = val
        if encoders:
            for col, encoder in encoders.items():
                if col not in input_df.columns:
                    continue
                val = input_df[col].iloc[0]
                if col in ('estimated_ride_time_min', 'actual_ride_time_min'):
                    try:
                        input_df[col] = encode_time_minutes(float(val), encoder)
                    except Exception:
                        input_df[col] = encode_time_minutes(55, encoder)
                    continue
                if isinstance(val, str):
                    try:
                        if val in encoder.classes_:
                            input_df[col] = encoder.transform([val])[0]
                        else:
                            input_df[col] = encoder.transform([encoder.classes_[0]])[0]
                    except Exception:
                        input_df[col] = 0
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        if scaler is not None:
            return scaler.transform(input_df)
        return input_df.values
    except Exception as e:
        st.error(f"Input preparation error: {e}")
        return None

def safe_day_label(val):
    dm = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
    try:
        return dm.get(int(float(str(val))), str(val))
    except Exception:
        return str(val)

def gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        domain={'x':[0,1],'y':[0,1]},
        title={'text':title,'font':{'color':'white'}},
        gauge={'axis':{'range':[0,100],'tickcolor':'white'},
               'bar':{'color':'#ff6b00'},
               'steps':[{'range':[0,30],'color':'#1a3a1a'},
                        {'range':[30,60],'color':'#3a3a1a'},
                        {'range':[60,100],'color':'#3a1a1a'}]}))
    fig.update_layout(paper_bgcolor='#1e1e1e', font_color='white', height=300)
    return fig

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;padding:20px 0;'>
            <div style='font-size:40px;'>🛵</div>
            <h2 style='color:#ff6b00;'>RAPIDO ML</h2>
            <p style='color:#888;font-size:12px;'>Intelligent Mobility Insights</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("Navigate", [
            "🏠 Home","📊 EDA Dashboard",
            "🎯 Ride Outcome Prediction","💰 Fare Prediction",
            "🚫 Cancellation Risk","⏱️ Driver Delay Prediction",
            "📈 Model Performance"
        ], label_visibility="collapsed")
        st.markdown("---")

        # CSV status
        st.markdown("### 📁 Data Status")
        if os.path.exists(CSV_PATH):
            size_mb = os.path.getsize(CSV_PATH)/1024/1024
            st.success(f"✅ CSV Ready ({size_mb:.1f} MB)")
        else:
            st.warning("⚠️ CSV not found (Git LFS may not have pulled)")

        st.markdown("### 🤖 Model Status")
        models = load_all_models()
        for num, name in {1:"Ride Outcome",2:"Fare Prediction",3:"Cancel Risk",4:"Driver Delay"}.items():
            if models[num]['model'] is not None:
                st.success(f"✅ Model {num}: {name}")
            else:
                st.error(f"❌ Model {num}: {name}")
        st.markdown("---")
        st.markdown("""<div style='text-align:center;color:#555;font-size:11px;'>
            Rapido ML v1.0<br>
            <a href='https://github.com/GeekyVishweshNeelesh/Rapido-Intelligent-Mobility-Insights_ML_Project'
               style='color:#ff6b00;'>⭐ GitHub Repo</a>
        </div>""", unsafe_allow_html=True)
    return page

def get_common_inputs(suffix="", model_num=1):
    defaults      = get_defaults(model_num)
    surge_default = defaults.get('surge_multiplier', 1.0)
    fare_default  = defaults.get('base_fare', 187.0)
    traffic_opts  = ["Low","Medium","High"]
    weather_opts  = ["Clear","Rain","Heavy Rain"]
    traffic_idx   = traffic_opts.index(defaults.get('traffic_level','Low')) if defaults.get('traffic_level','Low') in traffic_opts else 0
    weather_idx   = weather_opts.index(defaults.get('weather_condition','Clear')) if defaults.get('weather_condition','Clear') in weather_opts else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**🚗 Ride Details**")
        ride_distance_km        = st.number_input("Ride Distance (km)",        0.5,100.0,13.0,0.5, key=f"dist{suffix}")
        estimated_ride_time_min = st.number_input("Estimated Ride Time (min)", 3,  164,  25,  1,   key=f"est{suffix}")
    with c2:
        st.markdown("**⏰ Time & Location**")
        hour_of_day = st.slider("Hour of Day", 0, 23, 12, key=f"hour{suffix}")
        day_label   = st.selectbox("Day of Week",
                          ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],
                          index=2, key=f"day{suffix}")
        day_of_week = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(day_label)
        is_weekend  = 1 if day_of_week >= 5 else 0
        st.info(f"is_weekend: {'Yes ✅' if is_weekend else 'No'}")
        city = st.selectbox("City",["Bangalore","Chennai","Delhi","Hyderabad","Mumbai"],
                            index=2, key=f"city{suffix}")
    with c3:
        st.markdown("**🌤️ Conditions**")
        vehicle_type      = st.selectbox("Vehicle Type",     ["Auto","Bike","Cab"],       index=2, key=f"veh{suffix}")
        traffic_level     = st.selectbox("Traffic Level",    traffic_opts, index=traffic_idx, key=f"traf{suffix}")
        weather_condition = st.selectbox("Weather Condition",weather_opts, index=weather_idx, key=f"weath{suffix}")
        surge_multiplier  = st.number_input("Surge Multiplier", 1.0,5.0,surge_default,0.1, key=f"surge{suffix}")
        base_fare         = st.number_input("Base Fare (₹)",    10.0,500.0,fare_default,5.0, key=f"fare_in{suffix}")

    return {
        'ride_distance_km':ride_distance_km,
        'estimated_ride_time_min':estimated_ride_time_min,
        'actual_ride_time_min':estimated_ride_time_min,
        'hour_of_day':hour_of_day,'day_of_week':day_of_week,'is_weekend':is_weekend,
        'city':city,'vehicle_type':vehicle_type,'traffic_level':traffic_level,
        'weather_condition':weather_condition,'surge_multiplier':surge_multiplier,
        'base_fare':base_fare,'booking_value':round(base_fare*surge_multiplier,2),
    }

# ── PAGE 1: HOME ──
def page_home():
    st.markdown("<h1>🛵 Rapido: Intelligent Mobility Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;font-size:16px;'>Ride Patterns, Cancellations & Fare Forecasting — 100K Records, 4 ML Models</p>", unsafe_allow_html=True)
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("📊 Records",   "100,000","Bookings")
    with c2: st.metric("🤖 Models",    "4","15 Algorithms")
    with c3: st.metric("🔧 Features",  "115+","Engineered")
    with c4: st.metric("🎯 Benchmarks","3/4","Met")
    st.markdown("---")
    st.markdown("<h2>🏆 Model Performance Summary</h2>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    cards = [
        ("Model 1","Ride Outcome (XGBoost)",   "~87%", "Accuracy","✅ Meets",  "#00cc44"),
        ("Model 2","Fare Prediction (XGBoost)","9.92%","RMSE %",  "✅ Meets",  "#00cc44"),
        ("Model 3","Cancel Risk (LightGBM)",   "92.71%","Accuracy","✅ Exceeds","#00cc44"),
        ("Model 4","Driver Delay (LightGBM)",  "83.04%","Accuracy","⚠️ Close",  "#ff6b00"),
    ]
    for col,(t,s,v,m,st_,c_) in zip([c1,c2,c3,c4],cards):
        with col:
            st.markdown(f"""<div class='info-card'>
                <h3 style='color:#ff6b00;margin:0;'>{t}</h3>
                <p style='color:#888;margin:4px 0;font-size:12px;'>{s}</p>
                <h2 style='color:{c_};margin:0;'>{v}</h2>
                <p style='color:#888;margin:4px 0;'>{m}</p>
                <p style='color:{c_};margin:0;'>{st_}</p>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("<h2>💼 Business Use Cases</h2>", unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'><h3 style='color:#ff6b00;'>🎯 Reduce Cancellations by 20%</h3>
        <p>Model 3 flags high-risk bookings before they get cancelled.</p></div>
        <div class='info-card'><h3 style='color:#ff6b00;'>💰 Dynamic Fare Pricing</h3>
        <p>Model 2 estimates accurate fares from distance, traffic & surge.</p></div>
        <div class='info-card'><h3 style='color:#ff6b00;'>⏱️ Improve Driver ETA</h3>
        <p>Model 4 predicts delay-prone drivers for better allocation.</p></div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("<h2>🛠️ Tech Stack</h2>", unsafe_allow_html=True)
        for icon,name,desc in [("🐍","Python 3.10","Core language"),("🤖","Scikit-learn","Preprocessing & LR"),
                                ("⚡","XGBoost","Best for Outcome & Fare"),("💡","LightGBM","Best for Classification"),
                                ("🌐","Streamlit","Dashboard & Cloud Deploy"),("📊","Plotly","Interactive charts")]:
            st.markdown(f"""<div class='info-card' style='padding:8px 15px;'>
                <span style='font-size:18px;'>{icon}</span>
                <b style='color:#ff6b00;'> {name}</b>
                <span style='color:#888;font-size:12px;'> — {desc}</span>
            </div>""", unsafe_allow_html=True)

# ── PAGE 2: EDA (live from CSV via Git LFS) ──
def page_eda():
    st.markdown("<h1>📊 EDA Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("---")
    with st.spinner("📂 Loading data..."):
        df = load_master_data()

    if df is None:
        st.error("❌ CSV not found. Make sure `master_dataset_engineered.csv` was pulled via Git LFS.")
        st.code("git lfs pull", language="bash")
        return

    st.success(f"✅ Loaded {len(df):,} records | {len(df.columns)} columns")
    T = dict(paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e', font_color='white')
    tab1,tab2,tab3,tab4 = st.tabs(["🚗 Ride Patterns","🚫 Cancellations","💰 Fare Analysis","👥 Customer & Driver"])

    with tab1:
        st.markdown("<h2>🚗 Ride Patterns</h2>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            if 'booking_status' in df.columns:
                cnt = df['booking_status'].value_counts().reset_index()
                cnt.columns = ['Status','Count']
                fig = px.pie(cnt, values='Count', names='Status', title='Booking Status Distribution',
                             color_discrete_sequence=['#00cc44','#ff4444','#ff6b00','#4488ff'])
                fig.update_layout(**T)
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'vehicle_type' in df.columns:
                cnt = df['vehicle_type'].value_counts().reset_index()
                cnt.columns = ['Vehicle','Count']
                fig = px.bar(cnt, x='Vehicle', y='Count', title='Rides by Vehicle Type',
                             color='Count', color_continuous_scale='Oranges')
                fig.update_layout(**T)
                st.plotly_chart(fig, use_container_width=True)
        hour_col = next((c for c in ['hour_of_day','hour'] if c in df.columns), None)
        if hour_col:
            h = df.groupby(hour_col).size().reset_index(name='Rides').sort_values(hour_col)
            fig = px.line(h, x=hour_col, y='Rides', title='Rides by Hour of Day',
                          markers=True, color_discrete_sequence=['#ff6b00'])
            fig.update_layout(**T, xaxis_title='Hour')
            st.plotly_chart(fig, use_container_width=True)
        dow_col = next((c for c in ['day_of_week','weekday'] if c in df.columns), None)
        if dow_col:
            day_order = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
            df2 = df.copy()
            df2['_day'] = df2[dow_col].apply(safe_day_label)
            d = df2.groupby('_day').size().reset_index(name='Rides')
            d['_o'] = d['_day'].apply(lambda x: day_order.index(x) if x in day_order else 99)
            d = d.sort_values('_o').drop(columns='_o')
            fig = px.bar(d, x='_day', y='Rides', title='Rides by Day of Week',
                         color='Rides', color_continuous_scale='Oranges')
            fig.update_layout(**T, xaxis_title='Day')
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("<h2>🚫 Cancellation Analysis</h2>", unsafe_allow_html=True)
        df2 = df.copy()
        if 'booking_status' in df2.columns:
            df2['_cancel'] = df2['booking_status'].astype(str).str.lower().str.contains('cancel').astype(int)
        elif 'target_is_cancelled' in df2.columns:
            df2['_cancel'] = df2['target_is_cancelled']
        else:
            df2['_cancel'] = 0
        c1,c2 = st.columns(2)
        with c1:
            if hour_col:
                ch = df2.groupby(hour_col)['_cancel'].mean().reset_index()
                ch.columns = ['Hour','Rate']; ch['Rate'] *= 100
                fig = px.bar(ch, x='Hour', y='Rate', title='Cancellation Rate by Hour (%)',
                             color='Rate', color_continuous_scale='Reds')
                fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'vehicle_type' in df2.columns:
                cv = df2.groupby('vehicle_type')['_cancel'].mean().reset_index()
                cv.columns = ['Vehicle','Rate']; cv['Rate'] *= 100
                fig = px.bar(cv, x='Vehicle', y='Rate', title='Cancellation Rate by Vehicle (%)',
                             color='Rate', color_continuous_scale='Reds')
                fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)
        if 'traffic_level' in df2.columns:
            ct = df2.groupby('traffic_level')['_cancel'].mean().reset_index()
            ct.columns = ['Traffic','Rate']; ct['Rate'] *= 100
            fig = px.bar(ct, x='Traffic', y='Rate', title='Cancellation Rate by Traffic (%)',
                         color='Rate', color_continuous_scale='Oranges')
            fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)
        if 'weather_condition' in df2.columns:
            cw = df2.groupby('weather_condition')['_cancel'].mean().reset_index()
            cw.columns = ['Weather','Rate']; cw['Rate'] *= 100
            fig = px.bar(cw, x='Weather', y='Rate', title='Cancellation Rate by Weather (%)',
                         color='Rate', color_continuous_scale='Blues')
            fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("<h2>💰 Fare Analysis</h2>", unsafe_allow_html=True)
        fare_col = next((c for c in ['booking_value','base_fare','total_fare'] if c in df.columns), None)
        dist_col = next((c for c in ['ride_distance_km','ride_distance'] if c in df.columns), None)
        c1,c2 = st.columns(2)
        with c1:
            if fare_col:
                fd = df[fare_col].dropna(); fd = fd[(fd>0)&(fd<fd.quantile(0.99))]
                fig = px.histogram(fd, nbins=50, title=f'{fare_col} Distribution (₹)',
                                   color_discrete_sequence=['#ff6b00'])
                fig.update_layout(**T, xaxis_title='₹', yaxis_title='Count')
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if fare_col and 'vehicle_type' in df.columns:
                fv = df.groupby('vehicle_type')[fare_col].mean().reset_index()
                fv.columns = ['Vehicle','Avg Fare (₹)']
                fig = px.bar(fv, x='Vehicle', y='Avg Fare (₹)', title='Avg Fare by Vehicle Type',
                             color='Avg Fare (₹)', color_continuous_scale='Oranges')
                fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)
        if dist_col and fare_col:
            samp = df[[dist_col,fare_col]].dropna().sample(min(5000,len(df)))
            fig = px.scatter(samp, x=dist_col, y=fare_col, title='Distance vs Fare',
                             color_discrete_sequence=['#ff6b00'], opacity=0.4)
            fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("<h2>👥 Customer & Driver</h2>", unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            if 'cust_customer_age' in df.columns:
                fig = px.histogram(df['cust_customer_age'].dropna(), nbins=20,
                                   title='Customer Age Distribution',
                                   color_discrete_sequence=['#4488ff'])
                fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)
        with c2:
            if 'cust_customer_gender' in df.columns:
                gc = df['cust_customer_gender'].value_counts().reset_index()
                gc.columns = ['Gender','Count']
                fig = px.pie(gc, values='Count', names='Gender',
                             title='Customer Gender Distribution',
                             color_discrete_sequence=['#ff6b00','#4488ff','#00cc44'])
                fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)
        if 'payment_method' in df.columns:
            pm = df['payment_method'].value_counts().reset_index()
            pm.columns = ['Method','Count']
            fig = px.pie(pm, values='Count', names='Method', title='Payment Method Usage',
                         color_discrete_sequence=['#ff6b00','#4488ff','#00cc44','#ff4444'])
            fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)

# ── PAGE 3: RIDE OUTCOME ──
def page_ride_outcome():
    st.markdown("<h1>🎯 Ride Outcome Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Predict: Completed or Cancelled</p>", unsafe_allow_html=True)
    st.markdown("---")
    models = load_all_models()
    if models[1]['model'] is None:
        st.error("❌ Model 1 not loaded.")
        return
    st.markdown("<div class='section-header'>📝 Enter Booking Details</div>", unsafe_allow_html=True)
    inputs = get_common_inputs(suffix="_m1", model_num=1)
    st.markdown("**👤 Customer Profile**")
    ex1,ex2,ex3,ex4 = st.columns(4)
    with ex1: inputs['cust_total_bookings']      = st.number_input("Total Past Bookings",0,500,50, 1,   key="ctb_m1")
    with ex2: inputs['cust_completed_rides']     = st.number_input("Completed Rides",    0,500,45, 1,   key="ccr_m1")
    with ex3: inputs['cust_cancelled_rides']     = st.number_input("Cancelled Rides",    0,500, 2, 1,   key="ccan_m1")
    with ex4: inputs['cust_avg_customer_rating'] = st.number_input("Avg Rating",       1.0,5.0,4.5,0.1, key="crat_m1")
    st.markdown("**🚗 Driver Quality**")
    dr1,dr2,dr3 = st.columns(3)
    with dr1: inputs['driver_delay_rate']      = st.number_input("Driver Delay Rate",     0.0,1.0,0.02,0.01,key="ddr_m1")
    with dr2: inputs['driver_acceptance_rate'] = st.number_input("Driver Acceptance Rate",0.0,1.0,0.90,0.01,key="dar_m1")
    with dr3: inputs['driver_delay_count']     = st.number_input("Driver Delay Count",    0,50,  0,   1,    key="ddc_m1")
    st.markdown("---")
    if st.button("🎯 PREDICT RIDE OUTCOME"):
        if inputs['cust_total_bookings'] > 0:
            inputs['cust_cancellation_rate'] = (inputs['cust_cancelled_rides'] / inputs['cust_total_bookings']) * 100
        inputs['high_risk_pairing'] = 0
        inputs['surge_price_sensitive_cust'] = 0
        inputs['total_risk_factors'] = 0
        scaled = prepare_input(inputs, 1, models)
        if scaled is not None:
            with st.spinner("Predicting..."):
                pred  = models[1]['model'].predict(scaled)
                proba = models[1]['model'].predict_proba(scaled)
            # Confirmed from Colab: 0=Cancelled, 1=Completed
            class_names     = {0:"❌ Cancelled", 1:"✅ Completed"}
            predicted_class = int(pred[0])
            outcome = class_names.get(predicted_class, f"Class {predicted_class}")
            box     = "success-box" if predicted_class == 1 else "danger-box"
            st.markdown("---")
            st.markdown("<h2>📊 Prediction Result</h2>", unsafe_allow_html=True)
            c1,c2 = st.columns([1,2])
            with c1:
                st.markdown(f"<div class='prediction-box {box}'>{outcome}</div>", unsafe_allow_html=True)
                conf = float(proba[0][predicted_class]) * 100
                st.markdown(f"<div style='text-align:center;color:#888;margin-top:10px;'>Confidence: <b style='color:#ff6b00;'>{conf:.1f}%</b></div>", unsafe_allow_html=True)
            with c2:
                classes = models[1]['model'].classes_
                labels  = [class_names.get(int(c), str(c)) for c in classes]
                colors  = ['#ff4444' if 'Cancelled' in l else '#00cc44' for l in labels]
                fig = go.Figure(go.Bar(x=labels, y=[p*100 for p in proba[0]], marker_color=colors))
                fig.update_layout(title='Prediction Probabilities (%)',
                                  paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e',
                                  font_color='white', yaxis_title='Probability (%)')
                st.plotly_chart(fig, use_container_width=True)

# ── PAGE 4: FARE PREDICTION ──
def page_fare_prediction():
    st.markdown("<h1>💰 Fare Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Estimate accurate fare before trip confirmation</p>", unsafe_allow_html=True)
    st.markdown("---")
    models = load_all_models()
    if models[2]['model'] is None:
        st.error("❌ Model 2 not loaded.")
        return
    st.markdown("<div class='section-header'>📝 Enter Trip Details</div>", unsafe_allow_html=True)
    inputs = get_common_inputs(suffix="_m2", model_num=2)
    st.markdown("---")
    if st.button("💰 PREDICT FARE"):
        scaled = prepare_input(inputs, 2, models)
        if scaled is not None:
            with st.spinner("Calculating fare..."):
                predicted_fare = float(models[2]['model'].predict(scaled)[0])
            st.markdown("---")
            st.markdown("<h2>💰 Predicted Fare</h2>", unsafe_allow_html=True)
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class='prediction-box success-box'>
                    ₹ {predicted_fare:.2f}<br><small>Estimated Fare</small>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.metric("📉 Low Estimate",  f"₹ {predicted_fare*0.9:.2f}")
                st.metric("📈 High Estimate", f"₹ {predicted_fare*1.1:.2f}")
            with c3:
                d = inputs['ride_distance_km']; t = inputs['estimated_ride_time_min']
                st.metric("📏 Per KM",  f"₹ {predicted_fare/d:.2f}" if d>0 else "N/A")
                st.metric("⏱️ Per Min", f"₹ {predicted_fare/t:.2f}" if t>0 else "N/A")
            T = dict(paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e', font_color='white')
            breakdown = pd.DataFrame({'Component':['Base Fare','Surge Charges','Distance Fare'],
                                      'Amount (₹)':[inputs['base_fare'],
                                                    inputs['base_fare']*(inputs['surge_multiplier']-1),
                                                    inputs['ride_distance_km']*8]})
            fig = px.bar(breakdown, x='Component', y='Amount (₹)', color='Amount (₹)',
                         color_continuous_scale='Oranges', title='Fare Breakdown')
            fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)

# ── PAGE 5: CANCELLATION RISK ──
def page_cancellation_risk():
    st.markdown("<h1>🚫 Customer Cancellation Risk</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Predict probability a customer will cancel</p>", unsafe_allow_html=True)
    st.markdown("---")
    models = load_all_models()
    if models[3]['model'] is None:
        st.error("❌ Model 3 not loaded.")
        return
    st.markdown("<div class='section-header'>📝 Enter Customer & Booking Details</div>", unsafe_allow_html=True)
    inputs = get_common_inputs(suffix="_m3", model_num=3)
    st.markdown("**👤 Customer History**")
    ec1,ec2,ec3,ec4 = st.columns(4)
    with ec1: inputs['cust_customer_age']             = st.number_input("Customer Age",       18,80, 41,1,  key="cage_m3")
    with ec2: inputs['cust_customer_signup_days_ago'] = st.number_input("Account Age (days)", 0,3000,520,1, key="csignup_m3")
    with ec3: inputs['cust_total_bookings']           = st.number_input("Total Bookings",     0,500, 11,1,  key="ctotal_m3")
    with ec4: inputs['cust_completed_rides']          = st.number_input("Completed Rides",    0,500,  7,1,  key="ccr_m3")
    ec5,ec6 = st.columns(2)
    with ec5: inputs['cust_avg_customer_rating'] = st.number_input("Avg Rating",1.0,5.0,4.3,0.1,key="crat_m3")
    with ec6: inputs['cust_customer_gender']     = st.selectbox("Gender",["Male","Female","Non-Binary"],key="cgender_m3")
    st.markdown("---")
    if st.button("🚫 PREDICT CANCELLATION RISK"):
        if inputs['cust_total_bookings'] > 0:
            inputs['cust_cancellation_rate'] = ((inputs['cust_total_bookings']-inputs['cust_completed_rides'])/inputs['cust_total_bookings'])*100
        scaled = prepare_input(inputs, 3, models)
        if scaled is not None:
            with st.spinner("Analysing..."):
                proba = models[3]['model'].predict_proba(scaled)
            cancel_prob = float(proba[0][1]) * 100
            st.markdown("---")
            st.markdown("<h2>📊 Cancellation Risk Result</h2>", unsafe_allow_html=True)
            c1,c2 = st.columns([1,2])
            with c1:
                if cancel_prob < 30:
                    lbl,box,adv,fn = "🟢 LOW RISK",   "success-box","No special action needed.",              st.success
                elif cancel_prob < 60:
                    lbl,box,adv,fn = "🟡 MEDIUM RISK","warning-box","Send a reminder notification.",          st.warning
                else:
                    lbl,box,adv,fn = "🔴 HIGH RISK",  "danger-box", "Offer incentive or pre-confirm booking.",st.error
                st.markdown(f"""<div class='prediction-box {box}'>
                    {lbl}<br><br>Cancellation Probability:<br>
                    <span style='font-size:32px;'>{cancel_prob:.1f}%</span>
                </div>""", unsafe_allow_html=True)
                st.markdown("**💡 Recommendation:**")
                fn(adv)
            with c2:
                st.plotly_chart(gauge_chart(cancel_prob,"Cancellation Probability (%)"),use_container_width=True)

# ── PAGE 6: DRIVER DELAY ──
def page_driver_delay():
    st.markdown("<h1>⏱️ Driver Delay Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#888;'>Predict if a driver is likely to cause delays</p>", unsafe_allow_html=True)
    st.markdown("---")
    models = load_all_models()
    if models[4]['model'] is None:
        st.error("❌ Model 4 not loaded.")
        return
    st.markdown("<div class='section-header'>📝 Enter Driver & Trip Details</div>", unsafe_allow_html=True)
    inputs = get_common_inputs(suffix="_m4", model_num=4)
    st.markdown("**🚗 Driver Profile**")
    dp1,dp2,dp3,dp4 = st.columns(4)
    with dp1: inputs['driver_performance_tier']     = st.selectbox("Performance Tier",["Bronze","Silver","Gold","Platinum"],index=1,key="dtier_m4")
    with dp2: inputs['driver_experience_level']     = st.selectbox("Experience Level",["Newbie","Beginner"],index=1,key="dexp_m4")
    with dp3: inputs['driver_total_assigned_rides'] = st.number_input("Total Assigned Rides",0,1000,21,1,key="dtar_m4")
    with dp4: inputs['driver_delay_count']          = st.number_input("Past Delay Count",    0,100,  1,1,key="ddc_m4")
    st.markdown("---")
    if st.button("⏱️ PREDICT DRIVER DELAY"):
        scaled = prepare_input(inputs, 4, models)
        if scaled is not None:
            with st.spinner("Analysing..."):
                proba = models[4]['model'].predict_proba(scaled)
            delay_prob = float(proba[0][1]) * 100
            st.markdown("---")
            st.markdown("<h2>📊 Driver Delay Result</h2>", unsafe_allow_html=True)
            c1,c2 = st.columns([1,2])
            with c1:
                if delay_prob < 30:
                    lbl,box,adv,fn = "🟢 ON TIME",       "success-box","Reliable driver. Assign booking.",   st.success
                elif delay_prob < 60:
                    lbl,box,adv,fn = "🟡 POSSIBLE DELAY","warning-box","Monitor ETA. Notify customer.",       st.warning
                else:
                    lbl,box,adv,fn = "🔴 LIKELY DELAYED","danger-box", "High risk! Consider reassigning.",    st.error
                st.markdown(f"""<div class='prediction-box {box}'>
                    {lbl}<br><br>Delay Probability:<br>
                    <span style='font-size:32px;'>{delay_prob:.1f}%</span>
                </div>""", unsafe_allow_html=True)
                st.markdown("**💡 Recommendation:**")
                fn(adv)
            with c2:
                st.plotly_chart(gauge_chart(delay_prob,"Delay Probability (%)"),use_container_width=True)

# ── PAGE 7: MODEL PERFORMANCE ──
def page_model_performance():
    st.markdown("<h1>📈 Model Performance</h1>", unsafe_allow_html=True)
    st.markdown("---")
    T = dict(paper_bgcolor='#1e1e1e', plot_bgcolor='#1e1e1e', font_color='white')
    fallback = {
        1: pd.DataFrame({'Algorithm':['Logistic Regression','Random Forest','XGBoost','LightGBM'],
                         'Accuracy' :[0.72,0.85,0.87,0.86],'F1-Score':[0.71,0.84,0.86,0.85]}),
        2: pd.DataFrame({'Algorithm':['Linear Regression','Random Forest','XGBoost'],
                         'RMSE (%)' :[31.76,19.98,9.92],'R² Score':[0.69,0.88,0.9696]}),
        3: pd.DataFrame({'Algorithm':['Logistic Regression','Random Forest','XGBoost','LightGBM'],
                         'Accuracy' :[0.7324,0.9165,0.9252,0.9271],'AUC':[0.8138,0.9648,0.9837,0.9841]}),
        4: pd.DataFrame({'Algorithm':['Logistic Regression','Random Forest','XGBoost','LightGBM'],
                         'Accuracy' :[0.5818,0.6862,0.7938,0.8304],'AUC':[0.6188,0.7564,0.8908,0.9239]}),
    }
    summaries = {
        1:("XGBoost","~87% Accuracy",  "✅ MEETS Benchmark (85-90%)",  "success-box"),
        2:("XGBoost","9.92% RMSE",     "✅ MEETS Benchmark (≤10%)",    "success-box"),
        3:("LightGBM","92.71% Accuracy","✅ EXCEEDS Benchmark (85-90%)","success-box"),
        4:("LightGBM","83.04% Accuracy","⚠️ Close to Benchmark (83%)", "warning-box"),
    }
    tab1,tab2,tab3,tab4 = st.tabs(["🎯 Model 1","💰 Model 2","🚫 Model 3","⏱️ Model 4"])
    for tab,num in zip([tab1,tab2,tab3,tab4],[1,2,3,4]):
        with tab:
            loaded  = load_comparison_csv(num)
            df_comp = loaded if loaded is not None else fallback[num]
            c1,c2 = st.columns(2)
            with c1: st.dataframe(df_comp, use_container_width=True)
            with c2:
                y_col  = 'RMSE (%)' if num==2 else 'Accuracy'
                y_col  = y_col if y_col in df_comp.columns else df_comp.columns[1]
                target = 10 if num==2 else 0.85
                fig = px.bar(df_comp, x='Algorithm', y=y_col, color=y_col,
                             color_continuous_scale='Oranges', title=f'{y_col} Comparison')
                fig.add_hline(y=target, line_dash='dash', line_color='red',
                              annotation_text=f'Target: {target}')
                fig.update_layout(**T); st.plotly_chart(fig, use_container_width=True)
            best,metric,status,box = summaries[num]
            st.markdown(f"""<div class='prediction-box {box}' style='font-size:16px;'>
                🏆 Best: {best} &nbsp;|&nbsp; {metric} &nbsp;|&nbsp; {status}
            </div>""", unsafe_allow_html=True)

def main():
    page = render_sidebar()
    if   page == "🏠 Home":                    page_home()
    elif page == "📊 EDA Dashboard":           page_eda()
    elif page == "🎯 Ride Outcome Prediction": page_ride_outcome()
    elif page == "💰 Fare Prediction":         page_fare_prediction()
    elif page == "🚫 Cancellation Risk":       page_cancellation_risk()
    elif page == "⏱️ Driver Delay Prediction": page_driver_delay()
    elif page == "📈 Model Performance":       page_model_performance()

if __name__ == "__main__":
    main()
