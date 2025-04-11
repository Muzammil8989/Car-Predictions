# -*- coding: utf-8 -*-
"""Car Price Prediction Streamlit App"""

import streamlit as st
import pandas as pd
import pickle
import datetime as dt
import warnings

# Load model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = pickle.load(open('car_price_model.pkl', 'rb'))


# Custom CSS for modern styling
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    h1 {color: #2a3f5f;}
    .st-bw {background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);}
    .prediction {font-size: 1.4em; font-weight: bold; color: #2a3f5f;}
</style>
""", unsafe_allow_html=True)

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
            manufacturer = st.selectbox('Manufacturer', ['audi', 'bmw', 'chevrolet', 'honda', 'hyundai', 
                                                       'mercedes-benz', 'mitsubishi', 'nissan', 'opel', 
                                                       'renault', 'toyota', 'volkswagen'], index=4)
        with col2:
            model_name = st.text_input('Model Name', placeholder='e.g., Camry')

        col3, col4 = st.columns(2)
        with col3:
            prod_year = st.number_input('Production Year', min_value=1960, 
                                       max_value=dt.datetime.now().year, value=2020)
        with col4:
            category = st.selectbox('Category', ['jeep', 'sedan', 'van', 'vagon', 'coupe', 
                                                'hatchback', 'microbus', 'minivan', 'pickup'], index=1)

        # Technical Details
        st.header("⚙️ Technical Specifications")
        col5, col6 = st.columns(2)
        with col5:
            engine_volume = st.number_input('Engine Volume (L)', min_value=0.5, max_value=6.0, value=2.0)
            fuel_type = st.selectbox('Fuel Type', ['petrol', 'diesel', 'hybrid', 'cng', 'lpg'], index=1)
        with col6:
            mileage = st.number_input('Mileage (km)', min_value=0, value=50000, step=5000)
            cylinders = st.number_input('Cylinders', min_value=2, max_value=8, value=4)

        # Features & Comfort
        st.header("💺 Features & Comfort")
        col7, col8 = st.columns(2)
        with col7:
            leather_interior = st.selectbox('Leather Interior', ['yes', 'no'], index=1)
            gear_box_type = st.selectbox('Gear Box Type', ['automatic', 'manual', 'tiptronic'], index=0)
        with col8:
            airbags = st.number_input('Airbags', min_value=0, max_value=16, value=6)
            drive_wheels = st.selectbox('Drive Wheels', ['front', 'rear', '4x4'], index=0)

        # Exterior & Additional
        st.header("🎨 Exterior Details")
        col9, col10 = st.columns(2)
        with col9:
            doors = st.selectbox('Doors', [2, 4, 5], index=1)
            color = st.selectbox('Color', ['black', 'white', 'silver', 'gray', 'blue', 'red'], index=2)
        with col10:
            wheel = st.selectbox('Steering Wheel', ['left', 'right'], index=0)
        
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
            'Levy': 0  # Add Levy with default value
        }
        
        return pd.DataFrame(data, index=[0])

def preprocess_input(input_df):
    df = input_df.copy()
    
    # Convert binary features
    df['Leather interior'] = df['Leather interior'].map({'yes': 1, 'no': 0})
    df['Wheel'] = df['Wheel'].map({'left': 0, 'right': 1})
    
    # Create dummy variables for categorical columns
    categorical_cols = ['Drive wheels', 'Gear box type', 'Fuel type', 
                       'Manufacturer', 'Category', 'Color']
    
    for col in categorical_cols:
        dummies = pd.get_dummies(df[col], prefix=col.split()[0] if ' ' in col else col)
        df = pd.concat([df, dummies], axis=1)
    
    # List of all expected columns from the error message
    expected_columns = [
        'Levy', 'Manufacturer', 'Model', 'Prod. year', 'Category', 
        'Leather interior', 'Engine volume', 'Mileage', 'Cylinders',
        'Doors', 'Wheel', 'Color', 'Airbags', 'Turbo',
        'Drive_4x4', 'Drive_front', 'Drive_rear',
        'Gear_automatic', 'Gear_manual', 'Gear_tiptronic', 'Gear_variator',
        'Fuel_cng', 'Fuel_diesel', 'Fuel_hybrid', 'Fuel_hydrogen',
        'Fuel_lpg', 'Fuel_petrol', 'Fuel_plug-in hybrid'
    ]
    
    # Add missing columns with 0 values
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Convert all feature columns to numeric
    for col in expected_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Select and order columns exactly as expected
    df = df[expected_columns]
    
    return df

def format_pkr(pkr):
    """Formats PKR amount into lakh/crore notation with proper symbols"""
    pkr_rounded = round(pkr, 2)
    
    # Crore formatting (>= 1 crore)
    if pkr_rounded >= 10**7:  # 10,000,000 = 1 crore
        crore_value = pkr_rounded / 10**7
        return f"₨{crore_value:.2f} crore"
    
    # Lakh formatting (>= 1 lakh)
    elif pkr_rounded >= 10**5:  # 100,000 = 1 lakh
        lakh_value = pkr_rounded / 10**5
        return f"₨{lakh_value:.2f} lakh"
    
    # Regular formatting for smaller amounts
    else:
        return f"₨{pkr_rounded:,.2f}"

# Main app logic
input_df = user_input_features()
processed_df = preprocess_input(input_df)

if st.button('🚀 Get Price Prediction', use_container_width=True):
    try:
        prediction = model.predict(processed_df)
        usd_price = prediction[0]
        
        # Convert to PKR using State Bank of Pakistan rate
        EXCHANGE_RATE = 277.78  # PKR per 1 USD
        pkr_price = usd_price * EXCHANGE_RATE

        st.markdown("---")
        st.markdown("### 📈 Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estimated Value (USD)", f"${usd_price:,.2f}")
        with col2:
            # Format PKR with lakh/crore notation
            formatted_pkr = format_pkr(pkr_price)
            st.metric("Estimated Value (PKR)", formatted_pkr)
            
        st.caption(f"*Exchange rate: 1 USD = {EXCHANGE_RATE} PKR (State Bank of Pakistan)")

    except Exception as e:
        st.error(f"Error: {str(e)}")

# To run the app:
# streamlit run app.py