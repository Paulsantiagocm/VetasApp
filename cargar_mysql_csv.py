import pandas as pd
import mysql.connector
from mysql.connector import Error

def load_csv_to_mysql(csv_file_path, host, user, password, database, table_name):
    connection = None
    try:
        # Conectar al servidor MySQL (sin especificar la base de datos)
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )

        if connection.is_connected():
            cursor = connection.cursor()
            # Crear la base de datos si no existe
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
            connection.commit()

        # Reconectar especificando la base de datos
        connection = mysql.connector.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Leer el archivo CSV en un DataFrame de pandas
            data = pd.read_csv(csv_file_path)

            # Crear la tabla si no existe
            create_table_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                {', '.join([f'{col} VARCHAR(255)' for col in data.columns])}
            );
            """
            cursor.execute(create_table_query)
            connection.commit()

            # Insertar los datos del DataFrame en la tabla MySQL
            for _, row in data.iterrows():
                insert_row_query = f"""
                INSERT INTO {table_name} ({', '.join(data.columns)})
                VALUES ({', '.join(['%s'] * len(row))});
                """
                cursor.execute(insert_row_query, tuple(row))
            
            connection.commit()
            print(f"Datos cargados exitosamente en la tabla {table_name}")

    except Error as e:
        print(f"Error al conectar a MySQL: {e}")
    
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()
            print("Conexión a MySQL cerrada")

# Configuración
csv_file_path = '/teamspace/studios/this_studio/mysql_data.csv'  # Reemplaza con la ruta a tu archivo CSV
host = 'localhost'  # Reemplaza con tu host de MySQL
user = 'root'  # Reemplaza con tu usuario de MySQL
password = '12345678'  # Reemplaza con tu contraseña de MySQL
database = 'base_de_datos_transacciones'  # Reemplaza con el nombre de tu base de datos
table_name = 'tabla_de_datos_transacciones'  # Reemplaza con el nombre de la tabla

# Llamada a la función
load_csv_to_mysql(csv_file_path, host, user, password, database, table_name)
