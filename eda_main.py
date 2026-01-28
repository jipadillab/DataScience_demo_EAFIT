import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Dashboard EDA Interactivo", layout="wide", page_icon="📊")

def main():
    st.title("📊 Dashboard de Análisis Exploratorio de Datos (EDA)")
    st.markdown("""
    Sube uno de los archivos CSV generados (Economía, Energía o Medio Ambiente) 
    para realizar un análisis automático y visualización interactiva.
    """)

    # --- 1. CARGA DE DATOS ---
    with st.sidebar:
        st.header("1. Carga de Datos")
        uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])
        st.markdown("---")
        st.write("Creado por tu Asistente de IA")

    if uploaded_file is not None:
        # Cargar datos
        try:
            df = pd.read_csv(uploaded_file)
            
            # Intentar convertir columnas de fecha automáticamente
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except (ValueError, TypeError):
                        pass
            
            st.success(f"Archivo **{uploaded_file.name}** cargado exitosamente con {df.shape[0]} filas y {df.shape[1]} columnas.")
        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            return

        # --- 2. ANÁLISIS EXPLORATORIO (EDA) ---
        st.header("2. Resumen de Datos")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Filas", df.shape[0])
        with col2:
            st.metric("Total Columnas", df.shape[1])
        with col3:
            st.metric("Celdas Vacías", df.isna().sum().sum())

        with st.expander("🔍 Ver Dataframe y Estadísticas Descriptivas", expanded=True):
            st.dataframe(df.head())
            
            tab1, tab2 = st.tabs(["Estadísticas Numéricas", "Tipos de Datos"])
            with tab1:
                st.write(df.describe())
            with tab2:
                st.write(df.dtypes.astype(str))

        # --- 3. SISTEMA DE RECOMENDACIÓN Y VISUALIZACIÓN ---
        st.header("3. Visualización Inteligente")
        
        # Separar columnas por tipo
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        all_cols = df.columns.tolist()

        # Controles de Graficación
        c1, c2, c3 = st.columns(3)
        
        with c1:
            x_axis = st.selectbox("Eje X (Variable Independiente)", options=all_cols, index=0)
        with c2:
            y_axis = st.selectbox("Eje Y (Variable Dependiente)", options=['Ninguno'] + all_cols, index=0)
        with c3:
            color_var = st.selectbox("Agrupar/Color (Opcional)", options=['Ninguno'] + categorical_cols, index=0)

        # Lógica de Recomendación
        recomendacion = ""
        tipo_grafico_defecto = "Histograma"

        # Determinamos tipos de las selecciones
        x_is_num = x_axis in numeric_cols
        y_is_num = y_axis in numeric_cols and y_axis != 'Ninguno'
        x_is_cat = x_axis in categorical_cols
        x_is_date = x_axis in date_cols

        if x_is_date and y_is_num:
            recomendacion = "💡 Recomendación: Al usar fechas en X y números en Y, un **Gráfico de Línea** es ideal para ver tendencias."
            tipo_grafico_defecto = "Línea"
        elif x_is_cat and y_is_num:
            recomendacion = "💡 Recomendación: Para comparar categorías vs números, usa **Barras** o **Boxplot**."
            tipo_grafico_defecto = "Barras"
        elif x_is_num and y_is_num:
            recomendacion = "💡 Recomendación: Dos variables numéricas se visualizan mejor con un **Scatter Plot (Dispersión)**."
            tipo_grafico_defecto = "Dispersión"
        elif x_is_num and y_axis == 'Ninguno':
            recomendacion = "💡 Recomendación: Para ver la distribución de una sola variable numérica, usa un **Histograma**."
            tipo_grafico_defecto = "Histograma"
        
        st.info(recomendacion)

        # Selección manual de gráfico
        chart_type = st.radio("Selecciona tipo de gráfico:", 
                              ["Dispersión", "Línea", "Barras", "Histograma", "Boxplot", "Violin"],
                              index=["Dispersión", "Línea", "Barras", "Histograma", "Boxplot", "Violin"].index(tipo_grafico_defecto),
                              horizontal=True)

        # Generación del Gráfico
        fig = None
        color_arg = None if color_var == 'Ninguno' else color_var

        try:
            if chart_type == "Dispersión":
                fig = px.scatter(df, x=x_axis, y=None if y_axis == 'Ninguno' else y_axis, color=color_arg, title=f"{x_axis} vs {y_axis}")
            elif chart_type == "Línea":
                fig = px.line(df.sort_values(by=x_axis), x=x_axis, y=None if y_axis == 'Ninguno' else y_axis, color=color_arg, title=f"Tendencia de {y_axis} por {x_axis}")
            elif chart_type == "Barras":
                # Para barras, a veces es mejor agregar si hay muchos datos
                if y_axis != 'Ninguno':
                    fig = px.bar(df, x=x_axis, y=y_axis, color=color_arg, title=f"{y_axis} por {x_axis}")
                else:
                    fig = px.bar(df, x=x_axis, color=color_arg, title=f"Conteo de {x_axis}")
            elif chart_type == "Histograma":
                fig = px.histogram(df, x=x_axis, color=color_arg, nbins=30, title=f"Distribución de {x_axis}")
            elif chart_type == "Boxplot":
                fig = px.box(df, x=x_axis, y=None if y_axis == 'Ninguno' else y_axis, color=color_arg, title=f"Distribución de {x_axis}")
            elif chart_type == "Violin":
                fig = px.violin(df, x=x_axis, y=None if y_axis == 'Ninguno' else y_axis, color=color_arg, box=True, title=f"Densidad de {x_axis}")

            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # --- 4. EXPORTAR ---
            st.header("4. Exportar Datos Filtrados/Seleccionados")
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                st.markdown("**Descargar Gráfico:**")
                st.caption("Pasa el mouse sobre el gráfico y haz clic en el ícono de la cámara 📷 para descargar como PNG.")
            
            with col_exp2:
                st.markdown("**Descargar Datos:**")
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar CSV Actual",
                    data=csv,
                    file_name='datos_procesados.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.error(f"No se pudo generar el gráfico con la configuración actual. Intenta seleccionar un Eje Y válido. Error: {e}")

    else:
        st.info("Esperando archivo CSV. Por favor usa el panel lateral para cargar tus datos.")

if __name__ == "__main__":
    main()
