import os
from dotenv import load_dotenv
import pymysql
import pandas as pd

# Cargar variables del .env
load_dotenv()

# Obtener credenciales
DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
DB_PORT = int(os.getenv("DB_PORT", 3306))


# Función para conectar
def get_connection():
    return pymysql.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
    )

# Función para recuperar los datos de product
def get_products():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM products",conn)
    conn.close()
    return df

# Función para recuperar los datos de recolección diarios
def get_daily_stats():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM daily_product_stats",conn)
    conn.close()
    return df

# Función para recuperar los datos de supermarkets
def get_supermarkets():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM supermarkets", conn)
    conn.close()
    return df


# df = get_daily_stats()
