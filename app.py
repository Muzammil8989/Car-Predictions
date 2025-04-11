# -*- coding: utf-8 -*-
"""Car Price Prediction Streamlit App"""

import streamlit as st
import pandas as pd
import datetime as dt
import json
from xgboost import XGBRegressor

# Load model and feature names
@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model('car_price_model.json')
    return model

@st.cache_resource
def load_feature_names():
    with open('feature_names.json', 'r') as f:
        return json.load(f)

try:
    model = load_model()
    feature_names = load_feature_names()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()



# Create input fields
st.title('🚗 Smart Car Price Predictor')
st.markdown("#### 📊 Get instant market valuation for your vehicle")

def user_input_features():
    with st.container():
        st.markdown('<div class="st-bw">', unsafe_allow_html=True)
        
        # Vehicle Basics
        st.header("🔧 Vehicle Specifications")
        col1, col2 = st.columns(2)
        with col1:
            manufacturer = st.selectbox(
                'Manufacturer',
                options=['audi', 'bmw', 'chevrolet', 'honda', 'hyundai', 
                        'mercedes-benz', 'mitsubishi', 'nissan', 'opel', 
                        'renault', 'toyota', 'volkswagen'],
                index=4
            )
        with col2:
            model_name = st.text_input('Model Name', placeholder='e.g., Camry')

        col3, col4 = st.columns(2)
        with col3:
            prod_year = st.number_input(
                'Production Year', 
                min_value=1960, 
                max_value=dt.datetime.now().year, 
                value=2020
            )
        with col4:
            category = st.selectbox(
                'Category',
                options=['jeep', 'sedan', 'van', 'vagon', 'coupe', 
                        'hatchback', 'microbus', 'minivan', 'pickup'],
                index=1
            )

        # Technical Details
        st.header("⚙️ Technical Specifications")
        col5, col6 = st.columns(2)
        with col5:
            engine_volume = st.number_input(
                'Engine Volume (L)', 
                min_value=0.5, 
                max_value=6.0, 
                value=2.0, 
                step=0.1
            )
            fuel_type = st.selectbox(
                'Fuel Type',
                options=['petrol', 'diesel', 'hybrid', 'cng', 'lpg'],
                index=1
            )
        with col6:
            mileage = st.number_input(
                'Mileage (km)', 
                min_value=0, 
                value=50000, 
                step=5000
            )
            cylinders = st.number_input(
                'Cylinders', 
                min_value=2, 
                max_value=8, 
                value=4
            )

        # Features & Comfort
        st.header("💺 Features & Comfort")
        col7, col8 = st.columns(2)
        with col7:
            leather_interior = st.selectbox(
                'Leather Interior',
                options=['yes', 'no'],
                index=1
            )
            gear_box_type = st.selectbox(
                'Gear Box Type',
                options=['automatic', 'manual', 'tiptronic'],
                index=0
            )
        with col8:
            airbags = st.number_input(
                'Airbags', 
                min_value=0, 
                max_value=16, 
                value=6
            )
            drive_wheels = st.selectbox(
                'Drive Wheels',
                options=['front', 'rear', '4x4'],
                index=0
            )

        # Exterior & Additional
        st.header("🎨 Exterior Details")
        col9, col10 = st.columns(2)
        with col9:
            doors = st.selectbox(
                'Doors',
                options=[2, 4, 5],
                index=1
            )
            color = st.selectbox(
                'Color',
                options=['black', 'white', 'silver', 'gray', 'blue', 'red'],
                index=2
            )
        with col10:
            wheel = st.selectbox(
                'Steering Wheel',
                options=['left', 'right'],
                index=0
            )
        
        st.markdown('</div>', unsafe_allow_html=True)

        data = {
            'Manufacturer': manufacturer.lower(),
            'Model': model_name.lower(),
            'Prod. year': dt.datetime.now().year - prod_year,
            'Category': category.lower(),
            'Leather interior': leather_interior.lower(),
            'Fuel type': fuel_type.lower(),
            'Engine volume': engine_volume,
            'Mileage': mileage,
            'Cylinders': cylinders,
            'Gear box type': gear_box_type.lower(),
            'Drive wheels': drive_wheels.lower(),
            'Doors': doors,
            'Wheel': wheel.lower(),
            'Color': color.lower(),
            'Airbags': airbags,
            'Turbo': 1 if 'turbo' in str(engine_volume).lower() else 0,
            'Levy': 0
        }
        
        return pd.DataFrame(data, index=[0])

def preprocess_input(input_df):
    df = input_df.copy()
    
    # Convert binary features
    df['Leather interior'] = df['Leather interior'].map({'yes': 1, 'no': 0})
    df['Wheel'] = df['Wheel'].map({'left': 0, 'right': 1})
    
    # Create dummy variables
    categorical_cols = ['Drive wheels', 'Gear box type', 'Fuel type']
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col.split()[0] if ' ' in col else col)
        df = pd.concat([df, dummies], axis=1)
    
    # Ensure all required features exist
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    # Convert to numeric
    for col in feature_names:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df[feature_names]

def format_pkr(pkr):
    """Formats PKR amount into lakh/crore notation"""
    pkr_rounded = round(pkr, 2)
    if pkr_rounded >= 10**7:
        return f"₨{pkr_rounded/10**7:.2f} crore"
    elif pkr_rounded >= 10**5:
        return f"₨{pkr_rounded/10**5:.2f} lakh"
    return f"₨{pkr_rounded:,.2f}"

# Main app logic
input_df = user_input_features()

if st.button('🚀 Get Price Prediction', use_container_width=True):
    try:
        processed_df = preprocess_input(input_df)
        prediction = model.predict(processed_df)
        usd_price = prediction[0]
        
        # Convert to PKR
        EXCHANGE_RATE = 277.78
        pkr_price = usd_price * EXCHANGE_RATE

        st.markdown("---")
        st.markdown("### 📈 Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated Value (USD)", f"${usd_price:,.2f}")
        with col2:
            st.metric("Estimated Value (PKR)", format_pkr(pkr_price))
            
        st.caption(f"*Exchange rate: 1 USD = {EXCHANGE_RATE} PKR (State Bank of Pakistan)")
        
        st.info("""
        ℹ️ **Disclaimer**: Estimated value based on machine learning models. 
        Actual market price may vary.
        """)

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    <p>© 2023 Smart Car Price Predictor | Powered by XGBoost</p>
</div>
""", unsafe_allow_html=True)