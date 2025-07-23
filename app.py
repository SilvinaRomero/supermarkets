from db_connection.connect import get_products,get_supermarkets
from flask import Flask, render_template,request,jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
# modelo de predicción de oferta
with open("model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
tfidf = model_data["tfidf"]
le_special = model_data["le_special"]
le_category = model_data["le_category"]


@app.route('/')
def home():
    # Parámetros de filtro
    selected_supermarket = request.args.get('supermarket')
    selected_category = request.args.get('category')
    
    # Productos
    df = get_products()
    df = df.drop_duplicates(subset=['url'])

    # Filtrar por supermercado seleccionada
    if selected_supermarket and selected_supermarket.isdigit():
        df = df[df['supermarket_id'] == int(selected_supermarket)]
    
    # Filtrar por categoría seleccionada
    if selected_category and selected_category != '':
        df = df[df['category'] == selected_category]

    products = df.to_dict(orient='records')
    
    # Supermercados
    supermarkets_df = get_supermarkets()
    supermarkets = supermarkets_df.to_dict(orient='records')

    # Categorías manuales
    categories = ["leche", "cerveza", "yogur"]

    return render_template(
        'index.html',
        products=products,
        supermarkets=supermarkets,
        categories=categories,
        selected_supermarket=selected_supermarket,
        selected_category=selected_category
    )
    
@app.route('/predict', methods=['GET'])
def predict():
    product_id = request.args.get('product_id')
    day = request.args.get('day')
    df = get_products()
    product = df[df['id'] == int(product_id)]
    print('-'*50)
    print(product)
    print('-'*50)
    if product.empty:
        return jsonify({"prediction": "Producto no encontrado"})
    # Preprocesar igual que en entrenamiento
    supermarket_id = product['supermarket_id'].values[0]
    special_conditions = le_special.transform([product['special_conditions'].values[0].strip()])[0]
    category = le_category.transform([product['category'].values[0].strip()])[0]

    # TF-IDF para name
    name_tfidf = tfidf.transform([product['name'].values[0]]).toarray()
    # Crear el vector final igual que el modelo espera
    day = int(day)
    X_new = np.hstack(([supermarket_id, special_conditions, category, day], name_tfidf[0]))

    # Predicción real
    pred = model.predict([X_new])[0]
    print('-'*50)
    print(pred)
    print('-'*50)
    result = "Sí estará en oferta" if pred == 1 else "No estará en oferta"

    return jsonify({"prediction": result})
    

if __name__ == '__main__':
    app.run(debug=True)