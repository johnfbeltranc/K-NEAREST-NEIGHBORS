from utils import db_connect
engine = db_connect()

# Author: John Fredy Beltran Cuellar
# Date: 10/05/2025
# Goal: Clasificador de Vinos con KNN

# ========================================
# Step 0. Importar librerías
# ========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV

# ========================================
# Step 1. Cargar dataset (df_raw)
# ========================================
url = "https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/refs/heads/main/winequality-red.csv"
df_raw = pd.read_csv(url, sep=";")
df_raw.info()

df_raw.sample(10)

# Step 2. Preprocessing (df_baking)
# ========================================
df_baking = df_raw.copy()

# Limpieza de posibles comillas en los nombres de columnas
df_baking.columns = df_baking.columns.str.replace('"', '').str.strip()
df_baking.columns = df_baking.columns.str.replace(" ", "_")
df_baking = df_baking.drop(columns=["quality"])
df= df_baking.copy()
df.info()

# Step 3: EDA
df_train, df_test = train_test_split(df, test_size=0.1, random_state=2025)
df_train, df_val = train_test_split(df_train, test_size=0.15, random_state=2025)

# Reset index
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

df_train.shape, df_val.shape, df_test.shape

df.hist()
plt.tight_layout()
plt.show()

sns.pairplot(df, corner=True)

sns.heatmap(df_train.corr(), vmin=-1, vmax=1, annot=True, cmap='RdBu')
plt.show()

# Step 4. Machine Learning (KNN)
# ========================================
# -------------------------
# Target original
y = df_raw["quality"]

# Features
X_train = df_train.copy()
X_val = df_val.copy()
X_test = df_test.copy()
y_train = y.loc[X_train.index]
y_val = y.loc[X_val.index]
y_test = y.loc[X_test.index]


# Step 4b: Escalar datos
# -------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


# Step 4c: Entrenar KNN
# -------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Step 4d: Evaluar modelo
# -------------------------
y_pred_val = knn.predict(X_val_scaled)
y_pred_test = knn.predict(X_test_scaled)

print("Accuracy Validation:", accuracy_score(y_val, y_pred_val))
print("Accuracy Test:", accuracy_score(y_test, y_pred_test))

print("\nClassification Report (Validation):\n", classification_report(y_val, y_pred_val))
print("\nConfusion Matrix (Validation):\n", confusion_matrix(y_val, y_pred_val))


# Step 4e: Optimizar k con GridSearchCV
# ========================================
param_grid = {'n_neighbors': list(range(1, 21))}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train_scaled, y_train)

print("Mejor k encontrado:", grid.best_params_['n_neighbors'])
print("Mejor score CV:", grid.best_score_)


# Entrenar KNN con el mejor k
knn_best = KNeighborsClassifier(n_neighbors=grid.best_params_['n_neighbors'])
knn_best.fit(X_train_scaled, y_train)


# Evaluación del modelo optimizado
y_pred_val_best = knn_best.predict(X_val_scaled)
y_pred_test_best = knn_best.predict(X_test_scaled)

print("\nAccuracy Validation (mejor k):", accuracy_score(y_val, y_pred_val_best))
print("Accuracy Test (mejor k):", accuracy_score(y_test, y_pred_test_best))
print("\nClassification Report (Validation) con mejor k:\n", classification_report(y_val, y_pred_val_best))
print("\nConfusion Matrix (Validation) con mejor k:\n", confusion_matrix(y_val, y_pred_val_best))


# Step 5: Predicción con un ejemplo nuevo
# ========================================
nuevo_vino = np.array([[7.4, 0.7, 0, 1.9, 0.076, 11, 34, 0.9978, 3.51, 0.56, 9.4]])
nuevo_vino_scaled = scaler.transform(nuevo_vino)
prediccion = knn_best.predict(nuevo_vino_scaled)
print("Predicción calidad vino nuevo:", prediccion[0])


# Step 6: Visualización Predicciones vs Real
# ========================================
plt.scatter(x=y_pred_val_best, y=y_val, alpha=0.6)
plt.grid(True)
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.plot([min(y_val), max(y_val)], [min(y_val), max(y_val)], c='r', linestyle='--')
plt.title("Predicciones vs Valores Reales")
plt.show()

# Crear un DataFrame con predicciones y valores reales
df_plot = pd.DataFrame({
    'Predicciones': y_pred_val_best,
    'Valores_Reales': y_val
})

# Contar ocurrencias para cada combinación predicción/real
heatmap_data = df_plot.groupby(['Valores_Reales', 'Predicciones']).size().unstack(fill_value=0)

# Graficar heatmap
plt.figure(figsize=(8,6))
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Predicciones')
plt.ylabel('Valores Reales')
plt.title('Predicciones vs Valores Reales (Validation Set)')
plt.show()

# validacion de otros modelos
clf_tree = Pipeline([
    ('proc', MinMaxScaler()),
    ('tree', DecisionTreeRegressor(max_depth=6,random_state=2025))
])

clf_rf = Pipeline([
    ('proc', MinMaxScaler()),
    ('rf', RandomForestRegressor(random_state=2025))
])

clf_hb = Pipeline([
    ('proc', MinMaxScaler()),
    ('hb', HistGradientBoostingRegressor(random_state=2025))
])

param_grids = {
    'tree':{
        'tree__max_depth':[2,4,5,6,7],
        'tree__min_samples_split':[2,4,6],
    },
    'rf':{
        'rf__max_depth':[2,4,5,6,7],
        'rf__min_samples_split':[2,4,6],
        'rf__n_estimators':[50,100,150,200]
    },
    'hb':{
        'hb__max_depth':[2,4,5,6,7]
    }
}

models = [
          (clf_tree,'Decission Tree', 'tree'),
          (clf_rf,'Random Forest', 'rf'),
          (clf_hb,'Histogram GBoosting', 'hb')
]

performance = {}
for est, name, sname in models:
  print(est, name, sname)
  est.fit(X_train, y_train)
  estimator_cv = GridSearchCV(
      est,
      param_grid = param_grids[sname],
      cv = 5
  )
  estimator_cv.fit(X_train, y_train)
  y_hat = estimator_cv.predict(X_val)
  mse = round(mean_squared_error(y_val, y_hat))
  r2 = round(r2_score(y_val, y_hat), 2)
  best_params = estimator_cv.best_params_
  performance[name] = {
      'MSE': mse,
      'R2 Score': r2,
      'Best Params': best_params,
      'estimator': estimator_cv.best_estimator_
  }

  # Tree
rforest_reg_cv = RandomForestRegressor(max_depth=7, n_estimators=200,random_state=2025)
rforest_reg_cv.fit(X_train, y_train)
y_hat = rforest_reg_cv.predict(X_val)

print(f'RMSE: {np.sqrt(mean_squared_error(y_val, y_hat)):.2f}')
print(f'R2: {r2_score(y_val, y_hat):.2f}')

plt.scatter(x=y_hat, y=y_val)
plt.grid(True)
plt.xlabel('Predictions')
plt.ylabel('Real')
plt.plot([0, 80], [0, 80], c='r', linestyle='--', )
plt.show()

# Random Forest
# hb_reg_cv = HistGradientBoostingRegressor(max_depth=7)

hb_reg_cv.fit(X_train, y_train)
y_hat = hb_reg_cv.predict(X_val)

print(f'RMSE: {np.sqrt(mean_squared_error(y_val, y_hat)):.2f}')
print(f'R2: {r2_score(y_val, y_hat):.2f}') 

plt.scatter(x=y_hat, y=y_val)
plt.grid(True)
plt.xlabel('Predictions')
plt.ylabel('Real')
plt.plot([0, 80], [0, 80], c='r', linestyle='--', )
plt.show()

