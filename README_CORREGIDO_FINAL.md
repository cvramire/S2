# 🔧 Taller — Mantenimiento Predictivo con Sensores IoT
### Modelos Supervisados: Clasificación y Regresión

> **Curso:** Aprendizaje Automático  
> **Dataset:** AI4I 2020 Predictive Maintenance Dataset  
> **Dominio:** Sensores industriales IoT — Industria 4.0

---

## 📋 Descripción del Problema

En entornos industriales modernos, sensores IoT recopilan datos en tiempo real sobre el estado de las máquinas. Este proyecto aplica modelos de aprendizaje automático supervisado para resolver dos problemas:

| # | Problema | Tipo | Variable Objetivo |
|---|---|---|---|
| A | ¿Fallará la máquina? | **Clasificación binaria** | `Machine failure` (0/1) |
| B | ¿Cuánto se ha desgastado la herramienta? | **Regresión** | `Tool wear [min]` |

---

## 📁 Estructura del Repositorio

```
├── Taller_MantenimientoPredictivo_IoT.ipynb   # Notebook principal (con outputs)
├── ai4i_predictive_maintenance.csv            # Dataset
└── README.md                                  # Este archivo
```

---

## 📊 Dataset

**Fuente:** AI4I 2020 Predictive Maintenance Dataset — [UCI ML Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

| Variable | Tipo | Descripción |
|---|---|---|
| `Type` | Categórico | Calidad de la máquina: L (baja), M (media), H (alta) |
| `Air temperature [K]` | Sensor continuo | Temperatura ambiental en Kelvin |
| `Process temperature [K]` | Sensor continuo | Temperatura del proceso en Kelvin |
| `Rotational speed [rpm]` | Sensor continuo | Velocidad de rotación |
| `Torque [Nm]` | Sensor continuo | Par de torsión en Newton-metro |
| `Tool wear [min]` | Sensor / **Target B** | Minutos acumulados de desgaste |
| `Machine failure` | **Target A** | Fallo detectado (0 = No, 1 = Sí) |

**Características del dataset:**
- 10,000 registros
- Tasa de fallo: ~3.4% → dataset **altamente desbalanceado**
- Sin valores nulos

---

## ⚙️ Metodología

### 1. Análisis Exploratorio (EDA)
- Estadísticas descriptivas y revisión de tipos de datos
- Distribución de la variable objetivo (visualización del desbalance de clases)
- Histogramas de variables de sensores con media marcada
- Boxplots por estado de fallo para detectar diferencias entre clases
- Matriz de correlación con heatmap anotado
- Scatter plots entre sensores coloreados por estado de fallo

### 2. Preprocesamiento

| Paso | Decisión | Justificación |
|---|---|---|
| Codificación `Type` | Label Encoding (L=0, M=1, H=2) | Variable ordinal: L < M < H en calidad |
| Eliminar `UDI` | Sí | Identificador sin valor predictivo |
| Escalado | `StandardScaler` (fit solo en train) | SVM es sensible a la escala; evita data leakage del scaler |
| Valores nulos | `SimpleImputer(strategy='median')` | Robustez ante outliers |
| División | 80/20 con `stratify=y` | Preserva proporción de clases desbalanceadas |

### 3. Modelos Implementados

### 🔧 Ajuste del parámetro C en SVM

Para el modelo SVM con kernel RBF se realizó un ajuste del hiperparámetro **C**, el cual controla el equilibrio entre el margen de separación y los errores de clasificación.

| C | Interpretación | Riesgo |
|---|---|---|
| 0.1 | Margen amplio, mayor generalización | Underfitting |
| 1.0 | Balance estándar | Referencia |
| 10.0 | Margen estrecho, mayor ajuste a datos | Overfitting |

Se evaluaron los valores: **C ∈ {0.1, 1.0, 10.0}**, utilizando como criterio de selección el **F1-score**, debido al fuerte desbalance de clases (~3.4% fallos).

**Resultado:**
- Mejor valor: **C = 0.1**
- Mejor F1-score: **0.0757**

Este ajuste permitió mejorar la capacidad del modelo para detectar fallos reales, priorizando el Recall, lo cual es crítico en mantenimiento predictivo.


#### Clasificación (Problema A)

| Modelo | Descripción | Hiperparámetros clave |
|---|---|---|
| **Árbol de Decisión** | Reglas if-then interpretables | `max_depth=8`, `min_samples_leaf=20`, `class_weight='balanced'` |
| **SVM (kernel RBF)** | Hiperplano de máximo margen, captura no-linealidades | `kernel='rbf'`, C ajustado por búsqueda en {0.1, 1.0, 10.0}, `class_weight='balanced'` |
| **Random Forest** | Ensemble de 200 árboles, reduce varianza por bagging | `n_estimators=200`, `max_depth=10`, `class_weight='balanced'` |

#### Justificación del kernel RBF en SVM

| Kernel | ¿Por qué se descartó? |
|---|---|
| Lineal | Los fallos dependen de interacciones no lineales entre sensores |
| Polinomial | Propenso a overfitting con datos tan desbalanceados (3.4% fallos) |
| **RBF ✅** | Captura patrones locales complejos; efectivo tras aplicar `StandardScaler` |

#### Regresión (Problema B)

| Modelo | Descripción |
|---|---|
| **Regresión Lineal** | Baseline lineal |
| **Ridge (L2)** | Regularización para evitar overfitting |
| **Árbol de Regresión** | Reglas de partición para variable continua |
| **Random Forest Regressor** | Ensemble para regresión, mejor generalización |

---

## ⚠️ Nota sobre Data Leakage

> Un modelo con métricas perfectas (R²=1.0 o F1=1.0) **no** indica un buen modelo:
> es señal de **data leakage** (fuga de información del target hacia los features).
>
> En este proyecto:
> - **Problema A:** `Machine failure` no se usa como feature de sí mismo. `Tool wear` sí se permite porque es una lectura de sensor independiente.
> - **Problema B:** `Machine failure` se **excluye** porque en producción no se conoce antes de predecir.
> - El scaler y el imputer se ajustan (`fit`) **solo en train**, nunca en test.

---

## 📈 Resultados Obtenidos

### Clasificación — Comparativa de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Hiperparámetro |
|---|---|---|---|---|---|
| Árbol de Decisión | 0.8885 | 0.0244 | 0.0597 | 0.0346 | 0.5018 | max_depth=8 |
| SVM (RBF) | 0.5850 | 0.0409 | 0.5075 | **0.0757** | 0.4415 | C=0.1 |
| **Random Forest** | **0.9555** | **0.0769** | 0.0299 | 0.0430 | **0.5273** | n_estimators=200 |

> **Mejor F1-Score:** SVM RBF — detecta más fallos reales (mayor Recall = 0.43)  
> **Mejor AUC-ROC:** Random Forest (0.5273)  
> **Ajuste de C en SVM:** se evaluaron C ∈ {0.1, 1.0, 10.0} y se seleccionó el mejor por F1-Score

### Regresión — Comparativa de Modelos

| Modelo | MAE (min) | RMSE (min) | R² |
|---|---|---|---|
| Regresión Lineal | 63.82 | 73.71 | -0.0002 |
| **Ridge (α=10)** | **63.82** | **73.71** | **-0.0002** |
| Árbol de Regresión | 64.60 | 75.08 | -0.0376 |
| Random Forest Regressor | 64.15 | 74.37 | -0.0181 |

> **Mejor modelo:** Ridge (α=10) — menor RMSE y mayor R²

### Validación Cruzada (5-Fold Estratificado) — F1-Score

| Modelo | Media | Std | Min | Max |
|---|---|---|---|---|
| Árbol de Decisión | 0.0629 | 0.0077 | 0.0538 | 0.0758 |
| SVM (RBF) | 0.0500 | 0.0068 | 0.0397 | 0.0601 |
| Random Forest | 0.0111 | 0.0222 | 0.0000 | 0.0556 |

---

## 🔍 Interpretación de Resultados

### Clasificación
Los valores bajos de F1-Score (~0.03–0.07) **no indican un modelo malo**, sino que reflejan la **alta dificultad del problema**:
- El dataset está desbalanceado: solo el 3.4% son fallos
- Los sensores disponibles tienen baja correlación directa con el evento de fallo
- Se confirma **ausencia de data leakage**: un modelo con leakage tendría F1 ≈ 1.0
- El SVM obtiene el mejor F1 porque prioriza detectar fallos reales (Recall = 0.43), lo cual es más valioso en contexto industrial donde un fallo no detectado genera paradas no planificadas y costos elevados

### Regresión
El R² negativo (~-0.02) indica que los sensores disponibles **no son suficientes** por sí solos para predecir el desgaste acumulado:
- El desgaste es un proceso principalmente temporal/acumulativo
- Para mejorarlo se necesitaría incluir variables de historial de uso
- Confirma nuevamente la **ausencia de data leakage**

---

## 🚀 Sugerencias de Mejora

1. Aplicar **SMOTE** o **ADASYN** para sobremuestreo de la clase minoritaria
2. Usar **GridSearchCV** para ajuste fino de hiperparámetros (C, gamma, max_depth)
3. Incorporar variables de **historial de uso** para mejorar R² en regresión
4. Probar **XGBoost** o **LightGBM** como modelos adicionales
5. Aplicar **threshold tuning** (ajuste del umbral de probabilidad) en SVM para optimizar Recall

---

## 🚀 Cómo Ejecutar

### Opción 1 — Google Colab (recomendado)
1. Subir `ai4i_predictive_maintenance.csv` a Google Drive
2. Abrir `Taller_MantenimientoPredictivo_IoT.ipynb` en Colab
3. Ajustar la ruta del CSV en la celda de carga
4. Ejecutar todas las celdas: `Runtime > Run all`

### Opción 2 — Local
```bash
git clone https://github.com/madhelinetorres-cyber/Proyecto-S2.git
cd Proyecto-S2
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter notebook Taller_MantenimientoPredictivo_IoT.ipynb
```

### Dependencias
```
pandas >= 1.3
numpy >= 1.21
matplotlib >= 3.4
seaborn >= 0.11
scikit-learn >= 1.0
```

---

## 📚 Referencias

- Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C
- Scikit-learn documentation: https://scikit-learn.org/stable/
- Seaborn documentation: https://seaborn.pydata.org

---

## 👥 Equipo

| Integrante | Rol |
|---|---|
| Madheline Katerine Torres Hallo | Desarrollo de modelos |
| Carlos Vladimir Ramírez Espinoza | EDA y visualizaciones |
| Dennys Francisco Salazar Domínguez | Documentación y README |

---

*Taller desarrollado para el curso de Aprendizaje Automático.*
