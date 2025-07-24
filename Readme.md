# 🛍 Proyecto: Predicción en Supermercados

Este proyecto implementa un modelo de **Machine Learning** para predecir el comportamiento de productos en supermercados. Incluye la conexión a base de datos, procesamiento de datos, entrenamiento del modelo, evaluación y visualización de métricas.

---


## ⚙️ Configuración

1. Crear archivo `.env` en la raíz con:

DB_HOST=localhost
DB_USER=tu_usuario
DB_PASSWORD=tu_password
DB_NAME=supermarkets_db

---

## ▶️ Cómo usar el proyecto

### **1. Conectar la base de datos**
Archivo: `db_connection/connect.py`  

python db_connection/connect.py

---

### **2. Entrenar el modelo**
Ejecuta:

python model/model.py

Esto genera el archivo `model.pkl` con el modelo entrenado.

---

### **3. Levantar la API Flask**
Ejecuta:

python app.py

Por defecto, la API estará disponible en:
http://127.0.0.1:5000/

---

## 📌 Archivos principales

- `app.py` → API Flask
- `model/model.py` → Entrenamiento del modelo
- `model.pkl` → Modelo entrenado
- `db_connection/connect.py` → Conexión MySQL
- `templates/index.html` → Interfaz simple para predicciones
