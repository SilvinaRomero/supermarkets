import pandas as pd
from db_connection.connect import get_products
from db_connection.metrics_table import save_metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
import numpy as np


## cargar el dataset
df = get_products()
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.columns)

# --------------------------------------------------------#
# RANDOM FOREST - GRID SEARCH - CLASIFICACIÓN BINARIA
# --------------------------------------------------------#

## elegir variables apropiadas para predecir oferta
df = df[
    [
        "supermarket_id",
        "name",
        "price",
        "discount_price",
        "special_conditions",
        "category",
        "scraped_at",
    ]
]

## ahora comprobamos duplicados (con fechas)
df = df.drop_duplicates()
print(f"Duplicados:: ")
print(df.duplicated().sum)

## transformación de variables
# category -> objetivo
# special_conditions -> label_encoding

le_special = LabelEncoder()
le_category = LabelEncoder()
df["special_conditions"] = le_special.fit_transform(df["special_conditions"])
df["category"] = le_category.fit_transform(df["category"])

# añado el dia de la semana
df["scraped_at"] = pd.to_datetime(df["scraped_at"])
df["day_of_week"] = df["scraped_at"].dt.dayofweek

# Nueva columna: 1 si hay descuento, 0 si no
df["has_discount"] = (df["discount_price"] < df["price"]).astype(int)

# name -> vectorizar
tfidf = TfidfVectorizer(max_features=500)
X_name = tfidf.fit_transform(df["name"]).toarray()  # matriz TF-IDF

# Otras features
X_other = df[["supermarket_id", "special_conditions", "category", "day_of_week"]].values
# Unir todo
X = np.hstack((X_other, X_name))

# # Objetivo
y = df["has_discount"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42, class_weight="balanced")

param_rf = {
    "n_estimators": [10, 100, 500],
    "max_depth": [10, 30],
    "min_samples_split": [2, 3, 5],
    "criterion": ["entropy"],
}

# por el desbalanceo de clases, buscamos todas y definimos por f1 score.
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
}
grid = GridSearchCV(
    model, param_rf, cv=3, scoring=scoring, refit="f1"
)  # después de evaluar todas las métricas, se elige el mejor modelo según F1.
grid.fit(X_train, y_train)
best = grid.best_estimator_
print("Modelo óptimo:", best)


print("Mejores hiperparámetros:", grid.best_params_)

# Predicciones
y_pred = best.predict(X_test)

reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
print(reporte)
print("*" * 80)

# Generamos la fecha actual en formato YYYY-MM-DD
train_date = datetime.now().strftime("%Y-%m-%d")
## actualizamos la base de datos con las metricas obtenidas
metrics = {
    "model_name": "RandomForestClassifier",
    "accuracy": reporte["accuracy"],
    "precision_score": reporte["macro avg"]["precision"],  # Macro avg es mejor para desbalance
    "recall": reporte["macro avg"]["recall"],
    "f1_score": reporte["macro avg"]["f1-score"],
    "train_date": train_date,
    "train_size": len(X_train),
    "test_size": len(X_test)
}
print(metrics)
print("*" * 80)
## guardar metrica con fecha de entrenamiento
save_metrics(
    metrics["model_name"],
    metrics["accuracy"],
    metrics["precision_score"],
    metrics["recall"],
    metrics["f1_score"],
    metrics["train_date"],
    metrics["train_size"],
    metrics["test_size"]
)

# Guardar el modelo, vectorizador y label encoders
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": best,
        "tfidf": tfidf,
        "le_special": le_special,
        "le_category": le_category
    }, f)

print("✅ Modelo, TF-IDF y LabelEncoders guardados en model.pkl")

os.makedirs("images", exist_ok=True)

# matriz de confusión diaria
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusión")
date_str = datetime.now().strftime("%Y-%m-%d")
plt.savefig(f"images/confusion_matrix_{date_str}.png")
plt.close()
print(f"✅ Matriz de confusión guardada: confusion_matrix_{date_str}.png")


y_pred_prob = best.predict_proba(X_test)[:, 1]
# Generar y guardar curva ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.savefig("images/roc_curve.png", bbox_inches="tight")
plt.close()
print("✅ Curva ROC guardada en images/roc_curve.png")
