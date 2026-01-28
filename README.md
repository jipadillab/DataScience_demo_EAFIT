# 📊 Dashboard Interactivo de Análisis Exploratorio de Datos (EDA)

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

Este proyecto consta de dos componentes principales: un **Generador de Datos Sintéticos** (enfocado en temas de Colombia como economía, energía y medio ambiente) y un **Dashboard Interactivo en Streamlit** que permite realizar un Análisis Exploratorio de Datos (EDA) automático y visualización inteligente.

## 🚀 Características Principales

### 1. Generación de Datos Sintéticos
Script en Python capaz de crear datasets limpios (sin valores nulos) y complejos (datos numéricos, categóricos, lógicos, fechas) para:
* 🌱 **Agro-Economía:** Producción de cultivos, exportación y precios por departamento.
* ⚡ **Energía Renovable:** Plantas de generación, capacidad MW y estado operativo.
* lu **Monitoreo Ambiental:** Sensores de calidad de aire, PM2.5 y clima en ciudades principales.

### 2. Dashboard EDA (Streamlit)
* **Carga Dinámica:** Soporte para archivos CSV.
* **Detección de Tipos:** Identifica automáticamente columnas numéricas, categóricas y temporales.
* **Estadísticas Automáticas:** Muestra métricas clave, conteo de nulos y estadísticas descriptivas al instante.
* **Recomendación Inteligente de Gráficos:** El sistema sugiere el mejor gráfico (Dispersión, Línea, Barras, Histograma) basándose en las variables seleccionadas.
* **Exportación:** Permite descargar los gráficos generados (PNG) y los datos procesados (CSV).

## 📂 Estructura del Proyecto

```text
├── data_generator.py    # Script para crear los 3 datasets sintéticos (copiar código generador aquí)
├── main_eda.py          # Aplicación principal de Streamlit (Dashboard)
├── requirements.txt     # Librerías necesarias
├── README.md            # Documentación del proyecto
└── *.csv                # Archivos generados (agro_colombia.csv, etc.)
