import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

# ==============================
# CONFIGURACIÓN GLOBAL
# ==============================
RANDOM_SEED = 42
TEST_SIZE = 0.3
N_FEATURES = 10  # número de características a seleccionar
np.random.seed(RANDOM_SEED)

# ==============================
# 1. CARGA DE DATOS
# ==============================
df = pd.read_csv("creditcard.csv")

print("="*60)
print("ANÁLISIS EXPLORATORIO INICIAL")
print("="*60)
print(f"Shape original del dataset: {df.shape}")
print(f"Número de transacciones totales: {len(df):,}")
print(f"Número de fraudes: {df['Class'].sum():,}")
print(f"\nDistribución de clases:")
print(df["Class"].value_counts())
print(f"\nPorcentaje de fraudes: {df['Class'].mean()*100:.3f}%")
print(f"Ratio de desbalance: {(df['Class']==0).sum()/(df['Class']==1).sum():.0f}:1")

# ==============================
# 2. PREPROCESAMIENTO
# ==============================
print("\n" + "="*60)
print("PREPROCESAMIENTO")
print("="*60)

# Separar variables predictoras y variable objetivo
X = df.drop(columns=["Class"])
y = df["Class"]

# Escalado de todas las variables numéricas
print("\n1. Aplicando StandardScaler a todas las características...")
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print(f"   ✓ {X_scaled.shape[1]} características escaladas (media=0, std=1)")

# División en train/test (70/30)
print("\n2. División train/test (70%/30%) con estratificación...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED
)

print(f"   ✓ Train: {X_train.shape[0]:,} muestras")
print(f"   ✓ Test:  {X_test.shape[0]:,} muestras")
print(f"\nDistribución en conjunto de entrenamiento:")
print(f"  - Clase 0 (Normal): {(y_train==0).sum():,} ({(y_train==0).mean()*100:.2f}%)")
print(f"  - Clase 1 (Fraude): {(y_train==1).sum():,} ({(y_train==1).mean()*100:.2f}%)")
print(f"  - Ratio de desbalance: {(y_train==0).sum()/(y_train==1).sum():.0f}:1")

# ==============================
# 3. SELECCIÓN DE CARACTERÍSTICAS (SelectKBest)
# ==============================
print("\n" + "="*60)
print("SELECCIÓN DE CARACTERÍSTICAS")
print("="*60)
print(f"\n3. Aplicando SelectKBest con Mutual Information...")
print(f"   Técnica: Selección basada en tests estadísticos")
print(f"   Objetivo: Seleccionar las {N_FEATURES} características más predictivas")
print(f"   Métrica: Información Mutua (detecta relaciones no lineales)")
print(f"   Ventaja: Muy eficiente computacionalmente")
print(f"   IMPORTANTE: Se aplica ANTES del balanceo para evitar sesgo")
print("\n   ⏱️  Este proceso toma solo unos segundos...\n")

# SelectKBest con información mutua (mejor para clasificación)
# NOTA CRÍTICA: Se ajusta sobre datos SIN balancear para selección objetiva
selector = SelectKBest(score_func=mutual_info_classif, k=N_FEATURES)
selector.fit(X_train, y_train)

# Obtener características seleccionadas
selected_features = X_train.columns[selector.get_support()].tolist()

# Mostrar scores de las características seleccionadas
scores = pd.DataFrame({
    'Feature': X_train.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

print("\n" + "="*60)
print("VECTOR DE CARACTERÍSTICAS SELECCIONADAS")
print("="*60)
for i, feature in enumerate(selected_features, start=1):
    score = scores[scores['Feature'] == feature]['Score'].values[0]
    print(f"  {i:2d}. {feature:6s}  (Score: {score:.4f})")

# ==============================
# 4. BALANCEO DE CLASES (solo sobre entrenamiento con características seleccionadas)
# ==============================
print("\n" + "="*60)
print("ESTRATEGIA DE BALANCEO")
print("="*60)
print("\n4. Aplicando SMOTE (Synthetic Minority Over-sampling Technique)...")
print("   Estrategia: Sobremuestreo de la clase minoritaria (fraudes)")
print("   Nota: Se aplica DESPUÉS de selección de características")
print("   Nota: Se aplica SOLO sobre entrenamiento para evitar data leakage\n")

smote = SMOTE(random_state=RANDOM_SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train[selected_features], y_train)

print("Distribución DESPUÉS del balanceo:")
print(f"  - Clase 0 (Normal): {(y_train_bal==0).sum():,} ({(y_train_bal==0).mean()*100:.2f}%)")
print(f"  - Clase 1 (Fraude): {(y_train_bal==1).sum():,} ({(y_train_bal==1).mean()*100:.2f}%)")
print(f"  - Nuevo ratio: {(y_train_bal==0).sum()/(y_train_bal==1).sum():.1f}:1")
print("   ✓ Dataset balanceado correctamente")

# ==============================
# 5. GUARDAR RESULTADOS
# ==============================
print("\n" + "="*60)
print("RESUMEN Y GUARDADO DE ARCHIVOS")
print("="*60)

print(f"\n✓ Escalado:          StandardScaler aplicado")
print(f"✓ División:          70% train ({X_train.shape[0]:,}) / 30% test ({X_test.shape[0]:,})")
print(f"✓ Selección:         SelectKBest con Mutual Information")
print(f"✓ Características:   {N_FEATURES} seleccionadas de {X_scaled.shape[1]} originales")
print(f"✓ Balanceo:          SMOTE aplicado DESPUÉS de selección")

print(f"\nDimensiones finales:")
print(f"  - X_train_bal: {X_train_bal.shape}")
print(f"  - y_train_bal: {y_train_bal.shape}")
print(f"  - X_test:      {X_test[selected_features].shape}")
print(f"  - y_test:      {y_test.shape}")

pd.Series(selected_features).to_csv("selected_features.csv", index=False, header=False)
X_train_bal.to_csv("X_train_bal.csv", index=False)
y_train_bal.to_csv("y_train_bal.csv", index=False, header=True)
X_test[selected_features].to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False, header=True)

print("\n✓ Archivos guardados exitosamente:")
print("  - selected_features.csv")
print("  - X_train_bal.csv / y_train_bal.csv")
print("  - X_test.csv / y_test.csv")

print("\n" + "="*60)
print("✅ PREPROCESAMIENTO COMPLETADO CORRECTAMENTE")
print("="*60)