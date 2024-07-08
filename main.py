import pandas as pd
import numpy as np
from prefect import task, flow
import pymysql
from pymongo import MongoClient
import boto3
import re
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Parte 1: Extracción de datos

@task
def extraer_de_mysql():
    conexion = pymysql.connect(
        host='localhost',
        user='root',
        password='12345678',
        database='base_de_datos_transacciones'  
    )
    consulta = "SELECT * FROM tabla_de_datos_transacciones"
    df = pd.read_sql(consulta, conexion)
    conexion.close()
    return df

@task
def extraer_de_mongodb():
    cliente = MongoClient('mongodb://localhost:27017')
    db = cliente['base_de_datos_productos']
    coleccion = db['coleccion_productos']
    df = pd.DataFrame(list(coleccion.find()))
    return df

@task
def extraer_de_s3(nombre_balde, clave_archivo):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=nombre_balde, Key=clave_archivo)
    df = pd.read_csv(obj['Body'])
    return df

@task
def obtener_parametros_s3():
    nombre_balde = 'mybucketpaulcorellausfq'
    clave_archivo = 's3_file_data.csv'
    return nombre_balde, clave_archivo

# Parte 2: Integración y Manipulación de Datos 

@task
def limpiar_datos(datos_mysql, datos_mongodb, datos_s3):
    # Convertir 'sales_amount' a numérico y forzar errores a NaN
    datos_mysql['sales_amount'] = pd.to_numeric(datos_mysql['sales_amount'], errors='coerce')
    
    # Eliminar valores nulos
    datos_mysql.dropna(inplace=True)
    datos_mongodb.dropna(inplace=True)
    datos_s3.dropna(inplace=True)
    
    # Filtrar solo ventas con valores positivos
    datos_mysql = datos_mysql[datos_mysql['sales_amount'] > 0]
    datos_mongodb = datos_mongodb[datos_mongodb['price'] > 0]
    
    # Proponer nuevo criterio de limpieza: eliminar duplicados
    datos_mysql.drop_duplicates(inplace=True)
    datos_mongodb.drop_duplicates(inplace=True)
    datos_s3.drop_duplicates(inplace=True)
    
    return datos_mysql, datos_mongodb, datos_s3

@task
def preprocesar_datos(datos_mysql, datos_mongodb):
    scaler = MinMaxScaler()
    
    datos_mysql['sales_amount'] = scaler.fit_transform(datos_mysql[['sales_amount']])
    datos_mongodb['price'] = scaler.fit_transform(datos_mongodb[['price']])
    
    return datos_mysql, datos_mongodb

@task
def validar_datos(datos_mysql, datos_mongodb, datos_s3):
    patron_nombre_producto = re.compile(r'^[A-Za-z0-9_]+$')
    patron_fecha = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    
    # Validar nombres de productos
    nombres_productos_invalidos = datos_mongodb[~datos_mongodb['product_name'].apply(lambda x: bool(patron_nombre_producto.match(x)))]
    if not nombres_productos_invalidos.empty:
        print("Se encontraron nombres de productos inválidos en los datos de MongoDB:")
        print(nombres_productos_invalidos)
        raise ValueError("Hay nombres de productos inválidos en MongoDB")
    
    # Validar formato de fecha
    if not all(datos_mysql['transaction_date'].apply(lambda x: bool(patron_fecha.match(x)))):
        raise ValueError("Hay fechas de transacción inválidas en MySQL")
    
    return datos_mysql, datos_mongodb, datos_s3

# Parte 3: Entrenamiento de un modelo de regresión

class RedNeuronal(nn.Module):
    def __init__(self, input_size):
        super(RedNeuronal, self).__init__()
        self.layer_1 = nn.Linear(input_size, 64)
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.layer_1(x))
        x = self.relu(self.layer_2(x))
        x = self.layer_3(x)
        return x

@task
def entrenar_modelo(datos_mysql, datos_mongodb):
    # Convertir 'product_id' en ambos DataFrames a string para evitar conflictos al combinar
    datos_mysql['product_id'] = datos_mysql['product_id'].astype(str)
    datos_mongodb['product_id'] = datos_mongodb['product_id'].astype(str)
    
    # Combinar datos y seleccionar características y etiquetas
    datos = pd.merge(datos_mysql, datos_mongodb, left_on='product_id', right_on='product_id')
    print (datos)

    datos['transaction_date'] = pd.to_datetime(datos['transaction_date']).astype(int) / 10**9

    X = datos[['store_id', 'product_id', 'price', 'transaction_date']]
    y = datos['sales_amount']

    # Normalizar características numéricas
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Definir la red neuronal
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

    model = SalesPredictor()

    # Definir la función de pérdida y el optimizador
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento
    def train_model(model, X_train, y_train, epochs=100):
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            if (epoch+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    train_model(model, X_train, y_train)


    # Evaluación
    def evaluate_model(model, X_test, y_test):
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            loss = criterion(predictions, y_test)
            print(f'Test Loss: {loss.item():.4f}')

    evaluate_model(model, X_test, y_test)

    # Guardar el modelo entrenado
    torch.save(model.state_dict(), 'modelo_ventas.pth')
    # Guardar el modelo
    #torch.save(model.state_dict(), 'sales_predictor_model.pth')
    print("Modelo guardado como 'modelo_ventas.pth")
    
    return model

@flow(name="Extracción, Procesamiento y Entrenamiento de Datos")
def flujo_completo():
    datos_mysql = extraer_de_mysql()
    datos_mongodb = extraer_de_mongodb()
    nombre_balde, clave_archivo = obtener_parametros_s3()
    datos_s3 = extraer_de_s3(nombre_balde, clave_archivo)
        
    datos_mysql, datos_mongodb, datos_s3 = limpiar_datos(datos_mysql, datos_mongodb, datos_s3)
    datos_mysql, datos_mongodb, datos_s3 = validar_datos(datos_mysql, datos_mongodb, datos_s3)
    datos_mysql, datos_mongodb = preprocesar_datos(datos_mysql, datos_mongodb)
    
    modelo = entrenar_modelo(datos_mysql, datos_mongodb)
    
    return modelo

if __name__ == "__main__":
    flujo_completo()
