# Proyecto 2 — Detección de Fraude

**Curso:** INFO1185 - Inteligencia Artificial  
**Profesor:** Dr. Ricardo Soto Catalán  
**Integrantes:** Pablo Antivil Morales, Benjamín Espinoza

---

## Dataset

**Credit Card Fraud Detection** - [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transacciones
- 492 fraudes (0.172%)
- 30 características (Time, Amount, V1-V28)

---

## 1. Preprocesamiento

### 1.1 Escalado de Variables

Se aplicó `StandardScaler` a todas las características para normalizar los datos (media=0, std=1).

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Razón:** Los algoritmos k-NN y SVM son sensibles a la escala de las características.

### 1.2 División Train/Test (70%/30%)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.30,
    stratify=y,
    random_state=42
)
```

- Train: 199,364 muestras (344 fraudes)
- Test: 85,443 muestras (148 fraudes)
- `stratify=y`: Mantiene la proporción de clases en ambos conjuntos

---

## 2. Selección de Características

### Técnica: SelectKBest con Información Mutua

Se utilizó `SelectKBest` con `mutual_info_classif` para seleccionar las 10 características más relevantes.

```python
selector = SelectKBest(score_func=mutual_info_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

**Nota:** Se eligió este método en lugar de Sequential Forward Selection por su eficiencia computacional (segundos vs. horas) y capacidad para detectar relaciones no lineales.

### Vector de Características Seleccionadas

Las 10 características con mayor score de información mutua:

| # | Característica | Score |
|---|----------------|-------|
| 1 | V14 | 0.XXXX |
| 2 | V17 | 0.XXXX |
| 3 | V10 | 0.XXXX |
| 4 | V12 | 0.XXXX |
| 5 | V16 | 0.XXXX |
| 6 | V3 | 0.XXXX |
| 7 | V7 | 0.XXXX |
| 8 | V11 | 0.XXXX |
| 9 | V4 | 0.XXXX |
| 10 | V2 | 0.XXXX |

*Nota: Los scores exactos se encuentran en `selected_features.csv`*

---

## 3. Balanceo de Clases (SMOTE)

Se aplicó SMOTE para balancear el conjunto de entrenamiento.

```python
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_selected, y_train)
```

- Antes: 199,020 normales / 344 fraudes (578:1)
- Después: 199,020 normales / 199,020 fraudes (1:1)

**Importante:** SMOTE se aplicó solo al conjunto de entrenamiento. El test permanece desbalanceado para evaluación realista.

---

## Archivos Generados

- `selected_features.csv`: Características seleccionadas con scores
- `X_train_bal.csv`, `y_train_bal.csv`: Datos de entrenamiento balanceados
- `X_test.csv`, `y_test.csv`: Datos de prueba

---

## Referencias

- [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Scikit-learn: [Feature Selection](https://scikit-learn.org/stable/modules/feature_selection.html)
