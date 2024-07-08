import pandas as pd
import numpy as np

def generate_test_data():
    np.random.seed(42)

    # Datos para MySQL (transacciones de ventas)
    sql_data = {
        'transaction_id': np.arange(1, 101),
        'store_id': np.random.randint(1, 5, size=100),
        'product_id': np.random.randint(1, 20, size=100),
        'sales_amount': np.random.uniform(100, 200, size=100),
        'transaction_date': pd.date_range(start='1/1/2020', periods=100, freq='D')
    }
    sql_df = pd.DataFrame(sql_data)
    sql_df.to_csv('mysql_data.csv', index=False)

    # Datos para MongoDB (detalles de productos)
    nosql_data = [
        {'product_id': i, 'product_name': f'Producto_{i}',
        'category': f'Categoria_{i%5}', 'price': np.random.uniform(10,
        100)}
        for i in range(1, 21)
        ]
    pd.DataFrame(nosql_data).to_csv('mongodb_data.csv',index=False)

    # Datos para S3 (información de tiendas)
    s3_data = {
        'store_id': np.arange(1, 5),
        'store_name': [f'Tienda_{i}' for i in range(1, 5)],
        'location': ['Ubicacion_1', 'Ubicacion_2',
        'Ubicacion_3', 'Ubicacion_4']
        }
    s3_df = pd.DataFrame(s3_data)
    s3_df.to_csv('s3_data.csv', index=False)

generate_test_data()