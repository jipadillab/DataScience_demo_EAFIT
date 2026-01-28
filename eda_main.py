import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="EDA Pro - EAFIT",
    layout="wide",
    page_icon="🎓"
)

# --- PANEL LATERAL (SIDEBAR) ---
def configurar_sidebar():
    with st.sidebar:
        st.title("Configuración y Contexto")
        
        st.markdown("### 👨‍🏫 Autor")
        st.info("**Jorge Iván Padilla**\n\nDocente / Investigador\nUniversidad EAFIT")
        
        st.markdown("---")
        
        st.markdown("### 📚 ¿Qué es un EDA?")
        st.caption("""
        El **Análisis Exploratorio de Datos (EDA)** es una aproximación para analizar conjuntos de datos y resumir sus características principales, a menudo con métodos visuales.
        
        **Objetivos:**
        1. Descubrir patrones y anomalías.
        2. Probar hipótesis.
        3. Comprobar suposiciones.
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Filtros de Carga")
        
        # Slider para interactividad de muestras
        n_muestras = st.slider(
            "Cantidad de muestras a analizar:",
            min_value=10,
            max_value=1000,
            value=500,
            step=10,
            help="Reduce este número si el dataset es muy pesado para agilizar los gráficos."
        )
        return n_muestras

# --- FUNCIONES DE GRÁFICOS ---

def plot_correlation_heatmap(df):
    # Seleccionar solo columnas numéricas para correlación
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr = numeric_df.corr()
        fig = px.imshow(
            corr, 
            text_auto=True, 
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Mapa de Calor de Correlaciones (Pearson)"
        )
        return fig
    return None

def plot_radar_chart(df, cat_col, num_cols):
    """
    Genera un gráfico de Radar usando Graph Objects para evitar errores de 'shape' en Plotly Express.
    """
    if not num_cols or not cat_col:
        return None
    
    # 1. Agrupar y obtener promedios
    df_grouped = df.groupby(cat_col)[num_cols].mean().reset_index()
    
    # 2. Normalizar datos (Escala 0-1) para que el radar sea legible
    df_normalized = df_grouped.copy()
    for col in num_cols:
        max_val = df_grouped[col].max()
        min_val = df_grouped[col].min()
        if max_val != min_val:
            df_normalized[col] = (df_grouped[col] - min_val) / (max_val - min_val)
        else:
            df_normalized[col] = 0 # Evitar división por cero

    # 3. Construir el gráfico con Graph Objects (Más robusto)
    fig = go.Figure()

    for i, row in df_normalized.iterrows():
        # Valores de las variables
        r_values = row[num_cols].values.flatten().tolist()
        theta_values = list(num_cols)
        
        # Cerrar el ciclo del radar manualmente
        r_values += [r_values[0]]
        theta_values += [theta_values[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=r_values,
            theta=theta_values,
            fill='toself',
            name=str(row[cat_col])
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] # Como normalizamos, el rango es fijo
            )
        ),
        showlegend=True,
        title=f"Perfiles Promedio por {cat_col} (Datos Normalizados 0-1)"
    )
    return fig

# --- MAIN APP ---
def main():
    n_muestras_filter = configurar_sidebar()
    
    # Zona Central
    st.title("📈 Plataforma de Análisis Exploratorio de Datos (EDA)")
    st.markdown("#### Herramienta interactiva para la visualización y análisis estadístico.")
    
    # Carga de Archivo en el centro
    uploaded_file = st.file_uploader(
        "📂 Arrastra y suelta tu archivo CSV aquí (Economía, Energía, Medio Ambiente)", 
        type=["csv"],
        help="Sube archivos generados previamente."
    )

    if uploaded_file:
        try:
            # Cargar y aplicar filtro de muestras
            df_raw = pd.read_csv(uploaded_file)
            
            # Validar limites
            limit = min(n_muestras_filter, len(df_raw))
            df = df_raw.iloc[:limit].copy()
            
            # Conversión automática de fechas
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
            
            st.success(f"✅ Datos cargados: {len(df)} filas analizadas (de {len(df_raw)} totales).")

            # --- ESTRUCTURA DE PESTAÑAS ---
            tab1, tab2, tab3, tab4 = st.tabs([
                "📋 Resumen de Datos", 
                "📊 Análisis Univariado (Conteos/Pastel)", 
                "🔗 Relaciones y Correlaciones", 
                "🕸️ Análisis Avanzado (Radar/Jerarquía)"
            ])

            # PESTAÑA 1: RESUMEN
            with tab1:
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Filas", df.shape[0])
                col2.metric("Columnas", df.shape[1])
                col3.metric("Duplicados", df.duplicated().sum())
                col4.metric("Nulos Totales", df.isna().sum().sum())
                
                with st.expander("🔍 Ver Primeras Filas y Tipos de Datos", expanded=True):
                    st.dataframe(df.head(), use_container_width=True)
                    st.write("**Tipos de variables:**")
                    st.write(df.dtypes.astype(str).to_frame(name="Tipo de Dato").T)
                
                with st.expander("📉 Estadísticas Descriptivas"):
                    st.write(df.describe())

            # PESTAÑA 2: UNIVARIADO (Cualitativo y Atípicos)
            with tab2:
                st.subheader("Distribución de Variables")
                
                col_type = st.radio("¿Qué tipo de variable quieres analizar?", ["Cualitativa (Categórica)", "Cuantitativa (Numérica)"], horizontal=True)
                
                if col_type == "Cualitativa (Categórica)":
                    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns
                    if len(cat_cols) > 0:
                        selected_cat = st.selectbox("Selecciona Variable Categórica:", cat_cols)
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            # Gráfico de Barras (Conteos)
                            # Usamos reset_index para que plotly entienda bien las columnas
                            conteo = df[selected_cat].value_counts().reset_index()
                            conteo.columns = [selected_cat, 'Frecuencia']
                            fig_count = px.bar(conteo, x=selected_cat, y='Frecuencia', color=selected_cat, title=f"Frecuencia de {selected_cat}")
                            st.plotly_chart(fig_count, use_container_width=True)
                        with c2:
                            # Gráfico de Pastel
                            fig_pie = px.pie(df, names=selected_cat, title=f"Proporción de {selected_cat}", hole=0.3)
                            st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.warning("No hay variables categóricas en este dataset.")

                else: # Cuantitativa
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        selected_num = st.selectbox("Selecciona Variable Numérica:", num_cols)
                        
                        c1, c2 = st.columns(2)
                        with c1:
                            # Histograma
                            fig_hist = px.histogram(df, x=selected_num, nbins=30, title=f"Distribución de {selected_num}", marginal="box")
                            st.plotly_chart(fig_hist, use_container_width=True)
                        with c2:
                            # Boxplot para Atípicos
                            fig_box = px.box(df, y=selected_num, points="all", title=f"Análisis de Atípicos (Boxplot): {selected_num}")
                            st.plotly_chart(fig_box, use_container_width=True)
                    else:
                        st.warning("No hay variables numéricas.")

            # PESTAÑA 3: RELACIONES (Bivariado y Correlación)
            with tab3:
                st.subheader("Mapa de Correlaciones")
                fig_corr = plot_correlation_heatmap(df)
                if fig_corr:
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("Necesitas al menos 2 variables numéricas para correlaciones.")

                st.markdown("---")
                st.subheader("Análisis de Cruce de Variables (Scatter Plot)")
                
                c1, c2, c3 = st.columns(3)
                all_cols = df.columns
                x_axis = c1.selectbox("Eje X", all_cols, index=0)
                y_axis = c2.selectbox("Eje Y", all_cols, index=1 if len(all_cols)>1 else 0)
                color_var = c3.selectbox("Color (Agrupación)", ['Ninguno'] + list(df.columns))

                if x_axis and y_axis:
                    color_arg = None if color_var == 'Ninguno' else color_var
                    
                    # Verificar si podemos aplicar trendline (solo si ambos son numéricos)
                    is_numeric_x = pd.api.types.is_numeric_dtype(df[x_axis])
                    is_numeric_y = pd.api.types.is_numeric_dtype(df[y_axis])
                    
                    trend = "ols" if (is_numeric_x and is_numeric_y) else None
                    
                    try:
                        fig_scat = px.scatter(df, x=x_axis, y=y_axis, color=color_arg, 
                                              title=f"Relación: {x_axis} vs {y_axis}", 
                                              trendline=trend)
                        st.plotly_chart(fig_scat, use_container_width=True)
                    except Exception as e:
                        # Fallback si falla el trendline por tipos de datos raros
                        st.warning(f"No se pudo generar línea de tendencia: {e}")
                        fig_scat = px.scatter(df, x=x_axis, y=y_axis, color=color_arg, title=f"Relación: {x_axis} vs {y_axis}")
                        st.plotly_chart(fig_scat, use_container_width=True)

            # PESTAÑA 4: AVANZADO (Radar y Jerarquía)
            with tab4:
                st.subheader("🕸️ Gráfico de Radar (Multivariable)")
                st.caption("Compara categorías basándote en el promedio de múltiples variables numéricas.")
                
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                num_cols = df.select_dtypes(include=[np.number]).columns
                
                if len(cat_cols) > 0 and len(num_cols) > 2:
                    radar_cat = st.selectbox("Selecciona Categoría Principal (Ejes del Radar):", cat_cols)
                    radar_nums = st.multiselect("Selecciona Variables Numéricas (Métricas):", num_cols, default=list(num_cols)[:3])
                    
                    if radar_cat and len(radar_nums) >= 3:
                        fig_radar = plot_radar_chart(df, radar_cat, radar_nums)
                        if fig_radar:
                            st.plotly_chart(fig_radar, use_container_width=True)
                    else:
                        st.info("Selecciona al menos 3 variables numéricas para generar el radar.")
                else:
                    st.warning("Datos insuficientes para gráfico de radar (se requieren categóricas y múltiples numéricas).")

                st.markdown("---")
                st.subheader("☀️ Gráfico Sunburst (Jerarquías)")
                st.caption("Visualiza cómo se distribuyen los datos a través de múltiples niveles de categorías.")
                
                if len(cat_cols) >= 2:
                    sb_cols = st.multiselect("Selecciona orden de jerarquía (Anillo interior -> Exterior):", cat_cols, default=list(cat_cols)[:2])
                    if len(sb_cols) >= 2:
                        # Contar frecuencias por la jerarquía seleccionada
                        df_sun = df.groupby(sb_cols).size().reset_index(name='conteo')
                        fig_sun = px.sunburst(df_sun, path=sb_cols, values='conteo', title="Jerarquía de Datos")
                        st.plotly_chart(fig_sun, use_container_width=True)
                    else:
                        st.info("Selecciona al menos 2 categorías para el Sunburst.")
                else:
                    st.warning("Se necesitan al menos 2 columnas categóricas para este gráfico.")

        except Exception as e:
            st.error(f"Ocurrió un error crítico al procesar el archivo: {e}")
    else:
        # Mensaje de bienvenida
        st.info("👋 ¡Bienvenido! Por favor carga un archivo CSV en el panel central para comenzar.")
        
if __name__ == "__main__":
    main()
