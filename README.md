# Visión por Computador: QuantumViz

## 🧠 Trabajo 01: Fusión de Perspectivas - Registro de Imágenes y Medición del Mundo Real

### 📄 Descripción
Este repositorio contiene el folder **`QuantumViz_Vision-Computador_01`**, en el cual se encuentra una guía práctica para ejecutar, validar y entender el pipeline de registro y medición de objetos en imágenes. Diseñado para: reproducibilidad (notebooks), validación con datos sintéticos (ground-truth) y uso en proyectos académicos o muestra de portafolio.

- **Propósito:** detectar características, emparejar puntos, estimar transformaciones (homografías), fusionar vistas y medir objetos en la escena con calibración métrica.  
- **Entradas:** imágenes (p. ej. 3 vistas del comedor) o datasets sintéticos con transformaciones conocidas.  
- **Salidas:** imágenes registradas, figuras y medidas cuantificadas (CSV/JSON).

---

### 📁 Estructura del repositorio

```
proyecto-registro-imagenes/
├── README.md                        # (este archivo)
├── requirements.txt                 # dependencias Python
├── data/
│   ├── original/                    # imágenes reales o sintéticas del comedor
│   │   └── example_coords.json      # ejemplo de coordenadas medidas
│   └── synthetic/                   # dataset sintético para validación
│       ├── ground_truth.json
│       ├── results_summary.csv
│       └── transform_*/             # carpetas con transformaciones usadas
├── notebooks/                       # notebooks ejecutables (exploración, validación, pipeline)
├── results/
│   ├── figures/                     # figuras y visualizaciones generadas
│   └── measurements/                # salidas de medición (JSON, CSV)
└── src/                             # módulos Python del pipeline
		├── feature_detection.py         # detección y filtrado de keypoints
		├── matching.py                  # emparejamiento de descriptores y filtrado geométrico
		├── registration.py              # cálculo de homografías / transformaciones y fusión
		├── measurement.py               # calibración de escala y medidas en la escena
		└── utils.py                     # utilidades (I/O, visualización, helpers)
```

---

### 🎯 Objetivos del proyecto

1. Validar el pipeline usando transformaciones conocidas (data sintética).  
2. Registrar y fusionar múltiples vistas para generar un espacio de referencia común.  
3. Calibrar la escala (px → cm) usando objetos de referencia y medir nuevos objetos.  
4. Entregar notebooks reproducibles y un API mínimo para integrar el pipeline en scripts.

---

### 🧰 Instalación rápida

Se recomienda usar un entorno virtual.

**PowerShell (Windows)**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### ✅ (inputs / outputs / criterios de éxito)

- Inputs:
	- Carpeta 'data/original/' con las imágenes a registrar (típicamente 3 vistas).
	- (Opcional) 'data/synthetic/' con pares o secuencias transformadas y 'ground_truth.json' para validación.

- Outputs:
	- Imágenes registradas y visualizaciones en 'results/figures/'.
	- Mediciones y coordenadas guardadas en 'results/measurements/' ('coords.json', 'measurements.json').

- Criterios de éxito:
	- En validación sintética, el error de transformación frente al ground-truth es pequeño y las medidas calibradas están dentro del umbral esperado.

---

### ▶️ Uso — Notebooks (recomendado)

- notebooks/01_exploratory_analysis_enhanced.ipynb — inspección inicial y selección de puntos.
- notebooks/02_synthetic_validation.ipynb — validación con data/synthetic/ground_truth.json.
- notebooks/03_main_pipeline.ipynb — pipeline completo: detección → matching → registro → medición.

Abrir los notebooks con JupyterLab o VSCode. Los notebooks guardan resultados en results/ por defecto.

---

### 🧩 Uso — desde Python (ejemplo mínimo)

```
from src.feature_detection import detect_keypoints
from src.matching import match_descriptors
from src.registration import estimate_transform, warp_image
from src.measurement import calibrate_scale, measure_objects
```

#### ejemplo básico (pseudo-código)
```
img1 = load_image('data/original/view01.jpg')
img2 = load_image('data/original/view02.jpg')
kp1, des1 = detect_keypoints(img1)
kp2, des2 = detect_keypoints(img2)
matches = match_descriptors(des1, des2)
H, mask = estimate_transform(kp1, kp2, matches)
merged = warp_image(img2, H, reference=img1)
```

#### calibración y medición
```
scale = calibrate_scale(merged, ref_points, real_length_cm=117.0)
measures = measure_objects(merged, object_points, px_to_cm=scale)
save_results(measures, 'results/measurements/measurements.json')
```

---

### 🧠 Explicación técnica (resumen)

* Detección / descriptores: SIFT (preferible) con fallback a ORB cuando SIFT no está disponible.
* Emparejamiento: FLANN/BFMatcher + ratio test (p. ej. 0.75) + filtrado geométrico con RANSAC.
* Estimación de transformación: homografías (projective) entre vistas; composición de transformaciones para fusionar N vistas.
* Fusión / warp: remapeo a un lienzo común (mosaico o espacio de referencia) conservando resolución suficiente para medición.
* Calibración: usar objeto de referencia con dimensión conocida (por ejemplo, cuadro ancho = 117 cm), calcular px→cm por la distancia entre puntos de referencia.
* Métricas de validación: error angular (deg), traslación (px), factor de escala relativo y error absoluto en cm sobre medidas conocidas.

---

### ⚙️ Descripción breve de los módulos en 'src/'

- 'feature_detection.py': detecta keypoints y calcula descriptores (filtrado por respuesta y estabilidad).
- 'matching.py': empareja descriptores (p. ej. FLANN / BFMatcher), aplica ratio test y filtrado geométrico (RANSAC).
- 'registration.py': estima homografías o transformaciones projectivas entre vistas, compone transformaciones y genera una fusión/registro.
- 'measurement.py': funciones para calibrar escala usando objetos de referencia (medidas reales conocidas) y para calcular dimensiones de nuevos objetos.
- 'utils.py': I/O, helpers para visualización de correspondencias y utilidades comunes.

---

### 🔎 Validación sintética

Usa 'notebooks/02_synthetic_validation.ipynb' para:
- cargar 'data/synthetic/ground_truth.json';
- aplicar las transformaciones conocidas a imágenes base;
- ejecutar el pipeline y comparar parámetros estimados con el ground-truth (errores de rotación, traslación, escala).

---

### 📐 Escala y calibración

1. Identificar un objeto de referencia en la escena y su dimensión real (por ejemplo, ancho del cuadro = 117 cm).
2. Medir la distancia entre las esquinas en coordenadas de imagen/registro.
3. Calcular factor de conversión píxeles→centímetros y aplicar a las mediciones de otros objetos.

---

### ⚠️ Casos límite y recomendaciones

* Pocas correspondencias fiables: aumentar número de features, probar otros detectores/descriptores, o añadir vistas intermedias.
* Cambios radiométricos fuertes: aplicar preprocesado (ecualización CLAHE, histogram matching) antes del matching.
* Transformaciones extremas / oclusiones: comprobar inliers RANSAC; rechazar estimaciones con número de inliers insuficiente.
* Escala no homogénea: evitar extrapolar mediciones lejos del plano de referencia sin calibraciones adicionales (reconstrucción 3D o marcas adicionales).

---

### 📦 Tests y reproducibilidad

- Los notebooks sirven como pruebas reproducibles.
- Sugerencia: agregar tests unitarios en tests/ que validen: detección (>N keypoints), matching (>M matches), y que midan errores en data/synthetic/ bajo umbral.


