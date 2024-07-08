import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

class SalesPredictor(nn.Module):
    def __init__(self):
        super(SalesPredictor, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Cargar el modelo guardado
model = SalesPredictor()
model.load_state_dict(torch.load('modelo_ventas.pth'))
model.eval()

# Función para preprocesar la entrada del usuario
def preprocess_input(store_id, product_id, price, transaction_date):
    # Convertir la fecha a un formato numérico
    transaction_date = pd.to_datetime(transaction_date).timestamp()
    
    
    input_data = pd.DataFrame([[store_id, product_id, price, transaction_date]], 
                              columns=['store_id', 'product_id', 'price', 'transaction_date'])
    
    scaler = MinMaxScaler()
    input_data_scaled = scaler.fit_transform(input_data)
    
    # Convertir a tensor de PyTorch
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    
    return input_tensor

# Interfaz de usuario de Streamlit
st.title("Predicción de Ventas")
st.write("Ingrese los detalles para predecir las ventas:")

store_id = st.number_input("Store ID", min_value=1)
product_id = st.number_input("Product ID", min_value=1)
price = st.number_input("Price", min_value=0.0, format="%.2f")
transaction_date = st.date_input("Transaction Date", min_value=datetime(2000, 1, 1))

if st.button("Predecir"):
    input_tensor = preprocess_input(store_id, product_id, price, transaction_date)
    with torch.no_grad():
        prediction = model(input_tensor)
    st.write(f"Predicted Sales Amount: {prediction.item():.4f}")
