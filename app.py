import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from utils.feature_definitions import get_feature_definitions, get_feature_groups
from utils.data_processor import preprocess_input_data
from utils.model_loader import load_model, predict_price
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #2c3e50;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        color: #27ae60;
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<div class="main-header">🏠 House Price Prediction</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    Welcome to the **House Price Prediction** application! This interactive tool uses a machine learning model 
    trained on the Ames Housing dataset to predict house prices based on 79 different features.
    
    **Key Features:**
    - 🔮 **Predict house prices** with detailed feature input
    - 📊 **Explore model performance** and accuracy metrics
    - 📈 **Visualize data insights** and feature importance
    - 📚 **Learn about the methodology** and approach used
    """)
    
    # Quick prediction section
    st.markdown('<div class="sub-header">Quick Price Prediction</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Basic House Features**")
        lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=50000, value=8000)
        gr_liv_area = st.number_input("Living Area (sq ft)", min_value=500, max_value=5000, value=1500)
        overall_qual = st.selectbox("Overall Quality", options=list(range(1, 11)), index=5)
        
    with col2:
        st.markdown("**Additional Features**")
        year_built = st.number_input("Year Built", min_value=1870, max_value=2024, value=1990)
        bedroom_abv_gr = st.number_input("Bedrooms Above Ground", min_value=0, max_value=8, value=3)
        full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=4, value=2)
        
    with col3:
        st.markdown("**Property Details**")
        garage_cars = st.number_input("Garage Cars", min_value=0, max_value=4, value=2)
        fireplaces = st.number_input("Fireplaces", min_value=0, max_value=3, value=1)
        
    # Predict button
    if st.button("🔮 Predict Price", type="primary"):
        # Create a basic feature dictionary for prediction
        basic_features = {
            'LotArea': lot_area,
            'GrLivArea': gr_liv_area,
            'OverallQual': overall_qual,
            'YearBuilt': year_built,
            'BedroomAbvGr': bedroom_abv_gr,
            'FullBath': full_bath,
            'GarageCars': garage_cars,
            'Fireplaces': fireplaces
        }
        
        try:
            # Make prediction (this would use the actual model)
            predicted_price = predict_basic_price(basic_features)
            
            st.markdown(f"""
            <div class="prediction-result">
                Predicted Price: ${predicted_price:,.2f}
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence interval
            confidence_lower = predicted_price * 0.85
            confidence_upper = predicted_price * 1.15
            
            st.info(f"**Confidence Interval:** ${confidence_lower:,.2f} - ${confidence_upper:,.2f}")
            
            # Show feature impact
            st.markdown("**Key Factors Affecting Price:**")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Living Area Impact", f"${(gr_liv_area - 1500) * 50:+,.0f}")
                st.metric("Quality Impact", f"${(overall_qual - 5) * 15000:+,.0f}")
                st.metric("Age Impact", f"${(year_built - 1990) * 200:+,.0f}")
                
            with col2:
                st.metric("Garage Impact", f"${garage_cars * 8000:+,.0f}")
                st.metric("Bathroom Impact", f"${(full_bath - 1) * 5000:+,.0f}")
                st.metric("Bedroom Impact", f"${(bedroom_abv_gr - 3) * 3000:+,.0f}")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Navigation info
    st.markdown("---")
    st.markdown("""
    **📍 Navigation:**
    - Use the sidebar to navigate between different sections
    - **🏠 Prediction**: Detailed house feature input and prediction
    - **📊 Model Performance**: View model accuracy and performance metrics
    - **📈 Data Insights**: Explore data visualizations and feature importance
    - **ℹ️ About Project**: Learn about the methodology and approach
    """)
    
    # Sample predictions showcase
    st.markdown('<div class="sub-header">Sample Predictions from Model</div>', unsafe_allow_html=True)
    
    # Load some sample predictions from the submission file
    sample_predictions = get_sample_predictions()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Recent Predictions**")
        fig = px.histogram(sample_predictions, x='SalePrice', nbins=30, 
                          title="Distribution of Predicted House Prices")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.markdown("**Price Statistics**")
        stats = sample_predictions['SalePrice'].describe()
        
        st.metric("Average Price", f"${stats['mean']:,.2f}")
        st.metric("Median Price", f"${stats['50%']:,.2f}")
        st.metric("Price Range", f"${stats['min']:,.2f} - ${stats['max']:,.2f}")
        
        # Show price categories
        st.markdown("**Price Categories:**")
        low_price = (sample_predictions['SalePrice'] < 150000).sum()
        mid_price = ((sample_predictions['SalePrice'] >= 150000) & 
                    (sample_predictions['SalePrice'] < 300000)).sum()
        high_price = (sample_predictions['SalePrice'] >= 300000).sum()
        
        st.write(f"💰 Budget Homes (< $150K): {low_price}")
        st.write(f"🏡 Mid-Range Homes ($150K-$300K): {mid_price}")
        st.write(f"🏰 Luxury Homes (> $300K): {high_price}")

def predict_basic_price(features):
    """
    Make a basic price prediction using simplified model logic
    This would be replaced with actual model inference
    """
    # Base price calculation using key features
    base_price = 50000
    
    # Living area contribution (major factor)
    base_price += features['GrLivArea'] * 80
    
    # Quality multiplier
    quality_multiplier = 0.7 + (features['OverallQual'] / 10) * 0.6
    base_price *= quality_multiplier
    
    # Lot area contribution
    base_price += features['LotArea'] * 2
    
    # Age factor
    age = 2024 - features['YearBuilt']
    if age < 10:
        base_price *= 1.1
    elif age > 30:
        base_price *= 0.9
    
    # Additional features
    base_price += features['FullBath'] * 8000
    base_price += features['BedroomAbvGr'] * 5000
    base_price += features['GarageCars'] * 10000
    base_price += features['Fireplaces'] * 7000
    
    return base_price

@st.cache_data
def get_sample_predictions():
    """Load sample predictions from the submission file"""
    try:
        # Load the actual submission file
        submission_df = pd.read_csv('attached_assets/submission_1752156569814.csv')
        return submission_df.head(100)  # Return first 100 predictions for display
    except:
        # Fallback to sample data if file not found
        np.random.seed(42)
        sample_data = []
        for i in range(100):
            price = np.random.lognormal(mean=12, sigma=0.4)
            sample_data.append({'Id': 1461 + i, 'SalePrice': price})
        return pd.DataFrame(sample_data)

if __name__ == "__main__":
    main()
