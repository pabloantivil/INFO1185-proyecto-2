import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# ==============================
# CONFIGURACIÓN GLOBAL
# ==============================
RANDOM_SEED = 42
TEST_SIZE = 0.3
np.random.seed(RANDOM_SEED)

# ==============================
# 1. CARGA DE DATOS
# ==============================
df = pd.read_csv("creditcard.csv")

print("="*70)
print("ANÁLISIS EXPLORATORIO INICIAL")
print("="*70)
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
print("\n" + "="*70)
print("PREPROCESAMIENTO")
print("="*70)

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
# 3. EXPERIMENTO: JUSTIFICACIÓN DEL NÚMERO DE CARACTERÍSTICAS (k)
# ==============================
print("\n" + "="*70)
print("EXPERIMENTO: SELECCIÓN ÓPTIMA DE k")
print("="*70)
print("\nObjetivo: Determinar el número óptimo de características a seleccionar")
print("Método: Validación cruzada con diferentes valores de k")
print("Métrica: Recall (Sensibilidad) - prioritaria para detección de fraude")

# Valores de k a probar
k_values = [5, 10, 15, 20, 25]
results_k = []

# Clasificador base para evaluar (rápido y robusto)
base_clf = RandomForestClassifier(
    n_estimators=50, 
    max_depth=10,
    class_weight='balanced',  # Maneja desbalance sin SMOTE para esta evaluación
    random_state=RANDOM_SEED,
    n_jobs=-1
)

print("Evaluando k = ", end="", flush=True)
for k in k_values:
    print(f"{k}...", end=" ", flush=True)
    
    # Seleccionar k características
    selector_temp = SelectKBest(score_func=mutual_info_classif, k=k)
    X_train_k = selector_temp.fit_transform(X_train, y_train)
    
    # Validación cruzada estratificada (3-fold por eficiencia)
    cv_scores = cross_val_score(
        base_clf, X_train_k, y_train,
        cv=3,
        scoring='recall',  # Prioridad: detectar fraudes
        n_jobs=-1
    )
    
    results_k.append({
        'k': k,
        'recall_mean': cv_scores.mean(),
        'recall_std': cv_scores.std(),
        'cv_scores': cv_scores
    })

print("✓\n")

# Mostrar resultados
print("-"*70)
print("RESULTADOS: Desempeño por número de características")
print("-"*70)
print(f"{'k':<6} {'Recall (Media)':<18} {'Recall (Std)':<15} {'Evaluación'}")
print("-"*70)

best_result = max(results_k, key=lambda x: x['recall_mean'])
OPTIMAL_K = best_result['k']

for res in results_k:
    marker = " ← ÓPTIMO" if res['k'] == OPTIMAL_K else ""
    print(f"{res['k']:<6} {res['recall_mean']:.4f}           "
          f"{res['recall_std']:.4f}          {marker}")

print("\n" + "="*70)
print("JUSTIFICACIÓN DE k SELECCIONADO")
print("="*70)
print(f"\n✅ Valor óptimo: k = {OPTIMAL_K}")
print(f"\n📊 Razones:")
print(f"   1. Mayor sensibilidad promedio: {best_result['recall_mean']:.4f}")
print(f"   2. Desviación estándar aceptable: {best_result['recall_std']:.4f}")
print(f"   3. Reducción de dimensionalidad: {100*(1-OPTIMAL_K/30):.0f}%")
print(f"   4. Balance entre complejidad y desempeño")

# Guardar resultados del experimento
experiment_df = pd.DataFrame(results_k)[['k', 'recall_mean', 'recall_std']]
experiment_df.to_csv("k_selection_experiment.csv", index=False)
print(f"\n✓ Resultados guardados en: k_selection_experiment.csv")

# ==============================
# 4. SELECCIÓN DE CARACTERÍSTICAS CON k ÓPTIMO
# ==============================
print("\n" + "="*70)
print("SELECCIÓN DE CARACTERÍSTICAS (k={})".format(OPTIMAL_K))
print("="*70)
print(f"\n4. Aplicando SelectKBest con k={OPTIMAL_K}...")
print(f"   Técnica: Selección basada en tests estadísticos")
print(f"   Métrica: Información Mutua (detecta relaciones no lineales)")
print(f"   IMPORTANTE: Se aplica ANTES del balanceo para evitar sesgo\n")

# SelectKBest con k óptimo
selector = SelectKBest(score_func=mutual_info_classif, k=OPTIMAL_K)
selector.fit(X_train, y_train)

# Obtener características seleccionadas
selected_features = X_train.columns[selector.get_support()].tolist()

# Mostrar scores
scores = pd.DataFrame({
    'Feature': X_train.columns,
    'Score': selector.scores_
}).sort_values('Score', ascending=False)

print("="*70)
print("VECTOR DE CARACTERÍSTICAS SELECCIONADAS")
print("="*70)
for i, feature in enumerate(selected_features, start=1):
    score = scores[scores['Feature'] == feature]['Score'].values[0]
    print(f"  {i:2d}. {feature:6s}  (Score MI: {score:.4f})")

# ==============================
# 5. BALANCEO DE CLASES (SMOTE)
# ==============================
print("\n" + "="*70)
print("ESTRATEGIA DE BALANCEO")
print("="*70)
print("\n5. Aplicando SMOTE (Synthetic Minority Over-sampling Technique)...")
print("   Estrategia: Sobremuestreo de la clase minoritaria (fraudes)")
print("   ⚠️  CRÍTICO: Se aplica SOLO sobre entrenamiento (no validación ni test)")
print("   Nota: Se aplica DESPUÉS de selección de características\n")

smote = SMOTE(random_state=RANDOM_SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train[selected_features], y_train)

print("Distribución DESPUÉS del balanceo:")
print(f"  - Clase 0 (Normal): {(y_train_bal==0).sum():,} ({(y_train_bal==0).mean()*100:.2f}%)")
print(f"  - Clase 1 (Fraude): {(y_train_bal==1).sum():,} ({(y_train_bal==1).mean()*100:.2f}%)")
print(f"  - Nuevo ratio: {(y_train_bal==0).sum()/(y_train_bal==1).sum():.1f}:1")
print("   ✓ Dataset balanceado correctamente")

# ==============================
# GUARDAR RESULTADOS
# ==============================
print("\n" + "="*70)
print("RESUMEN Y GUARDADO DE ARCHIVOS")
print("="*70)

print(f"\n✓ Escalado:          StandardScaler aplicado")
print(f"✓ División:          70% train ({X_train.shape[0]:,}) / 30% test ({X_test.shape[0]:,})")
print(f"✓ Selección k:       Justificado experimentalmente (k={OPTIMAL_K})")
print(f"✓ Selección:         SelectKBest con Mutual Information")
print(f"✓ Características:   {OPTIMAL_K} seleccionadas de {X_scaled.shape[1]} originales")
print(f"✓ Balanceo:          SMOTE aplicado SOLO en train")

print(f"\nDimensiones finales:")
print(f"  - X_train_bal: {X_train_bal.shape}  ← CON datos sintéticos")
print(f"  - y_train_bal: {y_train_bal.shape}")
print(f"  - X_test:      {X_test[selected_features].shape}  ← SIN datos sintéticos")
print(f"  - y_test:      {y_test.shape}")

pd.Series(selected_features).to_csv("selected_features.csv", index=False, header=False)
X_train_bal.to_csv("X_train_bal.csv", index=False)
y_train_bal.to_csv("y_train_bal.csv", index=False, header=True)
X_test[selected_features].to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False, header=True)

print("\n✓ Archivos guardados exitosamente:")
print("  - k_selection_experiment.csv  ← NUEVO: justificación de k")
print("  - selected_features.csv")
print("  - X_train_bal.csv / y_train_bal.csv  ← CON SMOTE")
print("  - X_test.csv / y_test.csv            ← SIN SMOTE")

print("\n" + "="*70)
print("✅ PREPROCESAMIENTO COMPLETADO CORRECTAMENTE")
print("="*70)