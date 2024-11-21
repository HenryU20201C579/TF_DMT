import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Cargar el modelo y el scaler
@st.cache_resource
def load_model():
    with open('hotel_gb_model.pkl', 'rb') as gb:
        gb_model = pickle.load(gb)
    with open('hotel_scaler.pkl', 'rb') as sc:
        scaler = pickle.load(sc)
    with open('hotel_features.pkl', 'rb') as feat:
        features = pickle.load(feat)
    return gb_model, scaler, features

# Función para hacer predicciones
def predict_cancellation(data, model, scaler):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    probability = model.predict_proba(data_scaled)
    return prediction[0], probability[0]

def main():
    # Título de la aplicación
    st.title('Predicción de Cancelaciones de Reservas Hoteleras')
    st.subheader('Sistema de Predicción con Gradient Boosting')
    
    # Cargar modelo
    model, scaler, features = load_model()
    
    # Sidebar
    st.sidebar.header('Parámetros de la Reserva')
    
    def user_input_parameters():
        # Parámetros básicos
        hotel = st.sidebar.selectbox('Tipo de Hotel', ['Resort Hotel', 'City Hotel'], index=0)
        hotel_encoded = 0 if hotel == 'Resort Hotel' else 1
        
        lead_time = st.sidebar.slider('Tiempo de anticipación (días)', 0, 365, 30)
        
        arrival_month = st.sidebar.selectbox('Mes de llegada', 
            ['January', 'February', 'March', 'April', 'May', 'June', 
             'July', 'August', 'September', 'October', 'November', 'December'])
        month_encoded = {'January': 0, 'February': 1, 'March': 2, 'April': 3,
                        'May': 4, 'June': 5, 'July': 6, 'August': 7,
                        'September': 8, 'October': 9, 'November': 10, 'December': 11}[arrival_month]
        
        # Parámetros de estancia
        st.sidebar.subheader('Detalles de la Estancia')
        weekend_nights = st.sidebar.number_input('Noches de fin de semana', 0, 7, 1)
        week_nights = st.sidebar.number_input('Noches entre semana', 0, 14, 2)
        
        # Información del huésped
        st.sidebar.subheader('Información del Huésped')
        adults = st.sidebar.number_input('Número de adultos', 1, 4, 2)
        children = st.sidebar.number_input('Número de niños', 0, 3, 0)
        
        # Detalles de la reserva
        st.sidebar.subheader('Detalles de la Reserva')
        meal = st.sidebar.selectbox('Tipo de comida', 
                                  ['BB (Bed & Breakfast)', 'HB (Half Board)', 'FB (Full Board)', 'No Meal'])
        meal_encoded = {'BB (Bed & Breakfast)': 0, 'HB (Half Board)': 1, 
                       'FB (Full Board)': 2, 'No Meal': 3}[meal]
        
        market_segment = st.sidebar.selectbox('Segmento de mercado',
                                            ['Direct', 'Corporate', 'Online TA', 'Offline TA/TO', 'Groups', 'Aviation'])
        market_encoded = {'Direct': 0, 'Corporate': 1, 'Online TA': 2, 
                         'Offline TA/TO': 3, 'Groups': 4, 'Aviation': 5}[market_segment]
        
        repeated_guest = st.sidebar.checkbox('Cliente repetido')
        parking_spaces = st.sidebar.number_input('Espacios de parking requeridos', 0, 3, 0)
        special_requests = st.sidebar.number_input('Número de pedidos especiales', 0, 5, 0)
        
        data = {
            'hotel': hotel_encoded,
            'lead_time': lead_time,
            'arrival_date_month': month_encoded,
            'arrival_date_week_number': 1,
            'stays_in_weekend_nights': weekend_nights,
            'stays_in_week_nights': week_nights,
            'adults': adults,
            'children': children,
            'meal': meal_encoded,
            'country': 0,
            'market_segment': market_encoded,
            'distribution_channel': market_encoded,
            'is_repeated_guest': int(repeated_guest),
            'previous_cancellations': 0,
            'previous_bookings_not_canceled': 0,
            'reserved_room_type': 0,
            'assigned_room_type': 0,
            'booking_changes': 0,
            'deposit_type': 0,
            'days_in_waiting_list': 0,
            'adr': 100,
            'required_car_parking_spaces': parking_spaces,
            'total_of_special_requests': special_requests
        }
        
        features_df = pd.DataFrame(data, index=[0])
        return features_df
    
    # Capturar parámetros del usuario
    df = user_input_parameters()
    
    # Mostrar parámetros de entrada
    st.subheader('Detalles de la Reserva')
    st.write(df)
    
    # Predicción
    if st.button('Predecir Cancelación'):
        prediction, probability = predict_cancellation(df, model, scaler)
        
        # Mostrar resultado
        st.subheader('Resultado de la Predicción')
        
        # Crear columnas para organizar la visualización
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error('⚠️ Alta probabilidad de cancelación')
            else:
                st.success('✅ Baja probabilidad de cancelación')
                
        with col2:
            st.write(f'Probabilidad de cancelación: {probability[1]:.2%}')
        
        # Mostrar gráfico de probabilidad
        import plotly.graph_objects as go
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability[1] * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Probabilidad de Cancelación"},
            gauge = {
                'axis': {'range': [None, 100]},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}))
        
        st.plotly_chart(fig)
        
        # Factores de riesgo
        st.subheader('Factores de Riesgo')
        risk_factors = []
        if df['lead_time'].values[0] > 100:
            risk_factors.append("- Reserva realizada con mucha anticipación")
        if df['stays_in_weekend_nights'].values[0] + df['stays_in_week_nights'].values[0] > 7:
            risk_factors.append("- Estancia prolongada")
        if not df['is_repeated_guest'].values[0]:
            risk_factors.append("- Cliente nuevo")
        if df['market_segment'].values[0] in [2, 3]:  # Online y Offline TA
            risk_factors.append("- Reserva a través de agencia de viajes")
            
        if risk_factors:
            st.write("Factores que pueden aumentar el riesgo de cancelación:")
            for factor in risk_factors:
                st.write(factor)

if __name__ == '__main__':
    main()
