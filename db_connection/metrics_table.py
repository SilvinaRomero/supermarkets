from db_connection.connect import get_connection
import pandas as pd


# crear tabla para guardar las metricas de entrenamiento
def create_table_metrics():
    conn = get_connection()
    cursor = conn.cursor()
    query = """
    CREATE TABLE IF NOT EXISTS metrics (
        id INT AUTO_INCREMENT PRIMARY KEY,
        model_name VARCHAR(100) NOT NULL,
        accuracy FLOAT,
        precision_score FLOAT,
        recall FLOAT,
        f1_score FLOAT,
        train_date DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """

    cursor.execute(query)
    conn.commit()
    conn.close()
    cursor.close()
    print("✅ Tabla 'metrics' creada o ya existía.")


# modificar tabla, y añadir tamañno de entrenamiento test y prueba
def metrics_add_size():
    conn = get_connection()
    cursor = conn.cursor()
    query = """
        ALTER TABLE metrics
        ADD COLUMN train_size INT,
        ADD COLUMN test_size INT;
    """
    cursor.execute(query)
    conn.commit()
    conn.close()
    cursor.close()
    print("✅ Tabla 'metrics' actualizada con columnas train_size y test_size")


# insertar registros
def save_metrics(model_name, accuracy, precision, recall, f1_score, train_size, test_size):
    conn = get_connection()
    cursor = conn.cursor()

    query = """
    INSERT INTO metrics (model_name, accuracy, precision, recall, f1_score, train_size, test_size)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    """

    cursor.execute(query, (model_name, accuracy, precision, recall, f1_score, train_size, test_size))
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Métricas guardadas correctamente")



# create_table_metrics()
# metrics_add_size()
