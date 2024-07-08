import pandas as pd
from pymongo import MongoClient

# Ruta al archivo CSV
csv_file_path = '/teamspace/studios/this_studio/mongodb_data.csv'

# Leer el archivo CSV usando pandas
data = pd.read_csv(csv_file_path)

# Conectarse a MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['base_de_datos_productos']
collection = db['coleccion_productos']

# Convertir DataFrame de pandas a un diccionario
data_dict = data.to_dict(orient='records')

# Insertar los datos en la colección
collection.insert_many(data_dict)

print("Datos insertados exitosamente en MongoDB")