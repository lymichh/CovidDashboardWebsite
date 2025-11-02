from flask import Flask, render_template, jsonify, request
import pyodbc
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import json
import numpy as np

app = Flask(__name__)

# --- CONFIGURACIÓN DE LA CONEXIÓN ---
server = 'localhost,1433'
database = 'COVID_19'
username = 'project'
password = 'project123'

connection_string = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password}'
)

def run_query(query, params=None):
    try:
        db = pyodbc.connect(connection_string)
        cursor = db.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        db.close()
        return df
    except Exception as e:
        print(f"Error al ejecutar consulta: {e}")
        return pd.DataFrame()


# ------- QUERY 10: DASHBOARD DE KPIs MEJORADO -------
@app.route('/kpi_dashboard')
def kpi_dashboard():
    # 📊 QUERY 1: Estadísticas globales actuales
    query_global = """
        SELECT 
            SUM(Confirmed) as TotalConfirmados,
            SUM(Deaths) as TotalMuertes,
            SUM(Recovered) as TotalRecuperados,
            SUM(Active) as TotalActivos,
            COUNT(DISTINCT Country_Region) as PaisesAfectados
        FROM dbo.country_wise_latest
    """
    df_global = run_query(query_global)
    
    # 📊 QUERY 2: Tendencias semanales
    query_tendencias = """
        SELECT 
            SUM(_1_week_change) as CambioSemanal,
            SUM(Confirmed) as TotalActual,
            AVG(_1_week_increase) as PromedioAumentoSemanal
        FROM dbo.country_wise_latest
        WHERE _1_week_increase IS NOT NULL
    """
    df_tendencias = run_query(query_tendencias)
    
    # 📊 QUERY 3: Casos graves y testing
    query_testing = """
        SELECT 
            SUM(Serious_Critical) as CasosGraves,
            SUM(ActiveCases) as CasosActivosTotal,
            AVG(Tests_1M_pop) as PromedioTestsPorMillon,
            SUM(TotalTests) as TestsTotales
        FROM dbo.worldometer_data
        WHERE Tests_1M_pop IS NOT NULL
    """
    df_testing = run_query(query_testing)
    
    # 🧮 CALCULAR KPIs
    if not df_global.empty:
        total_confirmados = df_global['TotalConfirmados'].iloc[0]
        total_muertes = df_global['TotalMuertes'].iloc[0]
        total_recuperados = df_global['TotalRecuperados'].iloc[0]
        total_activos = df_global['TotalActivos'].iloc[0]
        paises_afectados = df_global['PaisesAfectados'].iloc[0]
        
        # KPI 1: Tasa de Recuperación (%)
        tasa_recuperacion = (total_recuperados / total_confirmados * 100) if total_confirmados > 0 else 0
        
        # KPI 2: Tasa de Mortalidad (%)
        tasa_mortalidad = (total_muertes / total_confirmados * 100) if total_confirmados > 0 else 0
        
        # KPI 3: Velocidad de Propagación (% cambio semanal)
        velocidad_propagacion = df_tendencias['PromedioAumentoSemanal'].iloc[0] if not df_tendencias.empty else 0
        
        # KPI 4: Carga Hospitalaria (% casos graves)
        casos_graves = df_testing['CasosGraves'].iloc[0] if not df_testing.empty else 0
        casos_activos_total = df_testing['CasosActivosTotal'].iloc[0] if not df_testing.empty else 1
        carga_hospitalaria = (casos_graves / casos_activos_total * 100) if casos_activos_total > 0 else 0
        
        # KPI 5: Tests por millón
        tests_por_millon = df_testing['PromedioTestsPorMillon'].iloc[0] if not df_testing.empty else 0
        
    else:
        # Valores por defecto si no hay datos
        tasa_recuperacion = 0
        tasa_mortalidad = 0
        velocidad_propagacion = 0
        carga_hospitalaria = 0
        paises_afectados = 0
        tests_por_millon = 0
        total_confirmados = 0
        total_muertes = 0
        total_recuperados = 0
    
    # 🎨 CREAR FIGURA CON SUBPLOTS
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
        ],
        subplot_titles=(
            '🎯 Tasa de Recuperación',
            '☠️ Tasa de Mortalidad',
            '📈 Velocidad de Propagación',
            '🏥 Carga Hospitalaria',
            '🌍 Países Afectados',
            '🧪 Tests por Millón'
        ),
        vertical_spacing=0.3,
        horizontal_spacing=0.15
    )
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🎯 GAUGE 1: TASA DE RECUPERACIÓN (SIN DELTA CONFUSO)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=tasa_recuperacion,
        title={
            'text': f"<b>Recuperación Global</b><br><span style='font-size:12px; color:gray'>{total_recuperados:,.0f} de {total_confirmados:,.0f} casos</span>",
            'font': {'size': 14}
        },
        number={'suffix': "%", 'font': {'size': 42, 'color': '#059669'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "lightgray", 'ticksuffix': '%'},
            'bar': {'color': "#059669", 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#10b981", 'width': 5},
                'thickness': 0.8,
                'value': tasa_recuperacion
            }
        }
    ), row=1, col=1)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ☠️ GAUGE 2: TASA DE MORTALIDAD
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=tasa_mortalidad,
        title={
            'text': f"<b>Mortalidad Global</b><br><span style='font-size:12px; color:gray'>{total_muertes:,.0f} fallecidos</span>",
            'font': {'size': 14}
        },
        number={'suffix': "%", 'font': {'size': 42, 'color': '#dc2626'}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 2, 'tickcolor': "lightgray", 'ticksuffix': '%'},
            'bar': {'color': "#dc2626", 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 2], 'color': '#d1fae5'},
                {'range': [2, 5], 'color': '#fef3c7'},
                {'range': [5, 10], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 5},
                'thickness': 0.8,
                'value': tasa_mortalidad
            }
        }
    ), row=1, col=2)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 📈 GAUGE 3: VELOCIDAD DE PROPAGACIÓN
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=abs(velocidad_propagacion),
        title={
            'text': "<b>Aumento Semanal</b><br><span style='font-size:12px; color:gray'>Promedio global</span>",
            'font': {'size': 14}
        },
        number={'suffix': "%", 'font': {'size': 42, 'color': '#f59e0b'}},
        gauge={
            'axis': {'range': [0, 50], 'tickwidth': 2, 'tickcolor': "lightgray", 'ticksuffix': '%'},
            'bar': {'color': "#f59e0b", 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 5], 'color': '#d1fae5'},
                {'range': [5, 15], 'color': '#fef3c7'},
                {'range': [15, 50], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#f97316", 'width': 5},
                'thickness': 0.8,
                'value': abs(velocidad_propagacion)
            }
        }
    ), row=1, col=3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🏥 GAUGE 4: CARGA HOSPITALARIA
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=carga_hospitalaria,
        title={
            'text': f"<b>Casos Graves / Activos</b><br><span style='font-size:12px; color:gray'>{casos_graves:,.0f} graves de {casos_activos_total:,.0f}</span>",
            'font': {'size': 14}
        },
        number={'suffix': "%", 'font': {'size': 42, 'color': '#8b5cf6'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "lightgray", 'ticksuffix': '%'},
            'bar': {'color': "#8b5cf6", 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#a855f7", 'width': 5},
                'thickness': 0.8,
                'value': carga_hospitalaria
            }
        }
    ), row=2, col=1)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌍 GAUGE 5: PAÍSES AFECTADOS (SIN GAUGE, SOLO NÚMERO)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig.add_trace(go.Indicator(
        mode="number",
        value=paises_afectados,
        title={
            'text': "<b>Expansión Global</b><br><span style='font-size:12px; color:gray'>Países con casos confirmados</span>",
            'font': {'size': 14}
        },
        number={
            'font': {'size': 70, 'color': '#1e40af', 'family': 'Arial Black'},
            'suffix': ' países'
        },
        domain={'x': [0, 1], 'y': [0.2, 0.8]}  # Ajustado para no tapar texto
    ), row=2, col=2)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🧪 GAUGE 6: TESTS POR MILLÓN (ARREGLADO - TEXTO VISIBLE)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=tests_por_millon,
        title={
            'text': "<b>Capacidad de Testeo</b><br><span style='font-size:12px; color:gray'>Promedio mundial</span>",
            'font': {'size': 14}
        },
        number={
            'font': {'size': 32, 'color': '#0891b2'},  # Tamaño reducido
            'suffix': "<br>tests/1M",
            'valueformat': ',.0f'
        },
        gauge={
            'axis': {
                'range': [0, 300000],
                'tickwidth': 2,
                'tickcolor': "lightgray",
                'tickformat': ',.0f',
                'ticksuffix': 'k',
                'tickvals': [0, 50000, 100000, 150000, 200000, 250000, 300000]
            },
            'bar': {'color': "#0891b2", 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 3,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 50000], 'color': '#fee2e2'},
                {'range': [50000, 150000], 'color': '#fef3c7'},
                {'range': [150000, 300000], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#06b6d4", 'width': 5},
                'thickness': 0.8,
                'value': tests_por_millon
            }
        }
    ), row=2, col=3)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🎨 LAYOUT GENERAL CON ANIMACIONES
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    fig.update_layout(
        title=dict(
            text='<b>📊 COVID-19 Dashboard de Indicadores Clave (KPIs)</b>',
            x=0.5,
            xanchor='center',
            font=dict(size=32, color='#1e3a8a', family='Inter, Arial')
        ),
        paper_bgcolor='#f8fafc',
        plot_bgcolor='#f8fafc',
        height=900,
        margin=dict(t=140, b=80, l=80, r=80),
        font=dict(family='Inter, Arial', size=13, color='#334155')
    )
    
    # Convertir a HTML con animaciones activadas
    graph_html = pio.to_html(fig, full_html=False, config={
        'displayModeBar': True,
        'displaylogo': False,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'covid_kpi_dashboard',
            'height': 1000,
            'width': 1600,
            'scale': 2
        },
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    })
    
    # 🎭 AGREGAR ANIMACIÓN CSS AL HTML
    graph_html = f"""
    <style>
        .plotly-graph-div {{
            animation: fadeIn 0.8s ease-in;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .gtitle {{ font-weight: 700 !important; }}
    </style>
    {graph_html}
    """
    
    return render_template('dashboard.html',
                         graph_html=graph_html,
                         titulo="Dashboard de Indicadores Clave",
                         tipo_grafico="kpi",
                         total_confirmados=f"{total_confirmados:,.0f}",
                         total_muertes=f"{total_muertes:,.0f}",
                         total_recuperados=f"{total_recuperados:,.0f}",
                         paises_afectados=paises_afectados)

# ------- QUERY 4: DISTRIBUCIÓN CONTINENTES CON HOVER AVANZADO -------
@app.route('/muerte_continente')
def muerte_continente():
    query_continentes = """
        SELECT 
            w.Continent as Continente, 
            SUM(c.Deaths) AS TotalMuertes
        FROM dbo.country_wise_latest c
        JOIN dbo.worldometer_data w
            ON c.Country_Region = w.Country_Region
        WHERE w.Continent IS NOT NULL
        GROUP BY w.Continent
        ORDER BY TotalMuertes DESC;
    """
    df_continentes = run_query(query_continentes)

    query_top_paises = """
        WITH RankedCountries AS (
            SELECT 
                w.Continent,
                c.Country_Region,
                c.Deaths,
                ROW_NUMBER() OVER (PARTITION BY w.Continent ORDER BY c.Deaths DESC) as rn
            FROM dbo.country_wise_latest c
            JOIN dbo.worldometer_data w
                ON c.Country_Region = w.Country_Region
            WHERE w.Continent IS NOT NULL
        )
        SELECT 
            Continent as Continente,
            Country_Region as PaisTop,
            Deaths as MuertesPaisTop
        FROM RankedCountries
        WHERE rn = 1
        ORDER BY Deaths DESC;
    """
    df_top_paises = run_query(query_top_paises)

    df_completo = df_continentes.merge(df_top_paises, on='Continente', how='left')

    total_mundial = df_completo['TotalMuertes'].sum()
    hover_texts = []
    for idx, row in df_completo.iterrows():
        porcentaje = (row['TotalMuertes'] / total_mundial * 100)
        ranking = idx + 1
        
        # Construir el texto del hover manualmente
        hover_text = (
            f"<b style='font-size:16px; color:#065f46;'>🌍 {row['Continente']}</b><br>"
            f"<br>"
            f"━━━━━━━━━━━━━━━━━━━━━<br>"
            f"<b>📊 Estadísticas Generales:</b><br>"
            f"💀 Total de muertes: <b style='color:#dc2626;'>{row['TotalMuertes']:,}</b><br>"
            f"📈 Porcentaje global: <b style='color:#059669;'>{porcentaje:.2f}%</b><br>"
            f"🏆 Ranking mundial: <b>#{ranking}</b><br>"
            f"<br>"
            f"━━━━━━━━━━━━━━━━━━━━━<br>"
            f"<b>🏴 País más afectado:</b><br>"
            f"📍 <b>{row['PaisTop']}</b><br>"
            f"💀 Muertes: <b style='color:#dc2626;'>{int(row['MuertesPaisTop']):,}</b><br>"
        )
        hover_texts.append(hover_text)

    # Colores
    colores = ['#065f46', '#047857', '#059669', '#10b981', '#34d399', '#6ee7b7']

    # Crear figura
    fig = go.Figure(data=[
        go.Pie(
            labels=df_completo['Continente'],
            values=df_completo['TotalMuertes'],
            hole=0.52,
            marker=dict(
                colors=colores[:len(df_completo)],
                line=dict(color='white', width=3)
            ),
            textinfo='label+percent',
            textposition='outside',
            textfont=dict(size=13, family='Inter', color='#333'),
            
            hovertext=hover_texts,
            hoverinfo='text', 
            
            hoverlabel=dict(
                bgcolor='white',
                font_size=13,
                font_family='Inter',
                bordercolor='#065f46',
                namelength=-1
            )
        )
    ])

    fig.update_layout(
        title=dict(
            text="☠️ Distribución Global de Muertes por COVID-19",
            x=0.5,
            xanchor='center',
            font=dict(size=26, color='#065f46', family='Inter', weight=700)
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            y=-0.15,
            x=0.5,
            xanchor='center',
            font=dict(size=13, family='Inter'),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#065f46',
            borderwidth=1
        ),
        annotations=[
            dict(
                text=f"<b style='font-size:18px;'>{total_mundial:,}</b><br>" +
                     f"<span style='font-size:12px;'>Muertes</span><br>" +
                     f"<span style='font-size:12px;'>totales</span>",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color='#065f46', family='Inter'),
                align='center'
            )
        ],
        paper_bgcolor='rgba(240,253,244,0.3)',
        plot_bgcolor='white',
        height=650,
        margin=dict(t=100, b=100, l=50, r=50),
        hovermode='closest'

    )

    graph_html = pio.to_html(fig, full_html=False, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d', 'zoomIn2d', 'zoomOut2d'],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'covid_muertes_continentes',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    })

    return render_template('dashboard.html',
                        graph_html=graph_html,
                        titulo="Distribución de Muertes por Continente",
                        tipo_grafico="pastel")

# ------- GRÁFICO DE BURBUJAS ANIMADO (VERSIÓN CORREGIDA) -------
@app.route('/burbujas_animado')
def burbujas_animado():
    query = """
        SELECT 
            Country_Region,Continent,Population,TotalCases,TotalDeaths,TotalRecovered,
            ActiveCases,Serious_Critical,Tot_Cases_1M_pop,Deaths_1M_pop,Tests_1M_pop,WHO_Region
        FROM dbo.worldometer_data
        WHERE 
            Continent IS NOT NULL AND Population > 0 AND TotalCases > 0
        ORDER BY TotalCases DESC
    """
    df = run_query(query)
    
    #LIMPIAR DATOS
    columnas_numericas = [
        'TotalDeaths', 'TotalRecovered', 'ActiveCases', 'Serious_Critical',
        'Deaths_1M_pop', 'Tests_1M_pop', 'Tot_Cases_1M_pop'
    ]
    
    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calcular tasas
    df['TasaMortalidad'] = df.apply(
        lambda row: (row['TotalDeaths'] / row['TotalCases'] * 100) if row['TotalCases'] > 0 else 0,
        axis=1
    ).round(2)
    
    df['TasaRecuperacion'] = df.apply(
        lambda row: (row['TotalRecovered'] / row['TotalCases'] * 100) if row['TotalCases'] > 0 else 0,
        axis=1
    ).round(2)
    
    df['PoblacionEscalada'] = (df['Population'] / 1_000_000).round(2)
    
    #COLORES
    colores_continentes = {
        'Asia': '#ef4444',
        'Europe': '#3b82f6',
        'Africa': '#f59e0b',
        'North America': '#10b981',
        'South America': '#8b5cf6',
        'Australia/Oceania': '#06b6d4'
    }
    df['Color'] = df['Continent'].map(colores_continentes)
    
    # 🎯 HOVER TEXT (VERSIÓN SEGURA)
    hover_texts = []
    for idx, row in df.iterrows():
        hover_text = (
            f"<b style='font-size:16px; color:#1e40af;'>🌍 {row['Country_Region']}</b><br>"
            f"<b>📍 Continente:</b> {row['Continent']}<br>"
            "<br>"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            "<b>📊 Estadísticas Absolutas:</b><br>"
            f"😷 Casos totales: <b>{int(row['TotalCases']):,}</b><br>"
            f"💀 Muertes totales: <b>{int(row['TotalDeaths']):,}</b><br>"
            f"💚 Recuperados: <b>{int(row['TotalRecovered']):,}</b><br>"
            f"🔥 Casos activos: <b>{int(row['ActiveCases']):,}</b><br>"
            f"🏥 Casos graves: <b>{int(row['Serious_Critical']):,}</b><br>"
            "<br>"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            "<b>📈 Estadísticas per Cápita:</b><br>"
            f"👥 Población: <b>{row['PoblacionEscalada']:.1f}M habitantes</b><br>"
            f"😷 Casos por millón: <b>{int(row['Tot_Cases_1M_pop']):,}</b><br>"
            f"💀 Muertes por millón: <b>{row['Deaths_1M_pop']:.1f}</b><br>"
            f"🧪 Tests por millón: <b>{int(row['Tests_1M_pop']):,}</b><br>"
            "<br>"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            "<b>📊 Tasas Calculadas:</b><br>"
            f"☠️ Tasa de mortalidad: <b>{row['TasaMortalidad']:.2f}%</b><br>"
            f"💚 Tasa de recuperación: <b>{row['TasaRecuperacion']:.2f}%</b>"
        )
        hover_texts.append(hover_text)

    
    fig = go.Figure()
    
    for continente in df['Continent'].unique():
        df_continente = df[df['Continent'] == continente]
        
        fig.add_trace(go.Scatter(
            x=df_continente['TotalCases'],
            y=df_continente['TotalDeaths'],
            mode='markers',
            name=continente,
            marker=dict(
                size=df_continente['PoblacionEscalada'],
                color=df_continente['Color'],
                opacity=0.7,
                line=dict(color='white', width=2),
                sizemode='diameter',
                sizeref=2.*max(df['PoblacionEscalada'])/(60.**2),
                sizemin=4
            ),
            text=df_continente['Country_Region'],
            hovertext=df_continente.apply(lambda row: hover_texts[df.index.get_loc(row.name)], axis=1),
            hoverinfo='text',
            hoverlabel=dict(
                bgcolor='white',
                font_size=13,
                font_family='Inter',
                bordercolor=df_continente['Color'].iloc[0]
            )
        ))
    
    # LÍNEA DE TENDENCIA
    df_validos = df[(df['TotalCases'] > 0) & (df['TotalDeaths'] > 0)].copy()
    
    if len(df_validos) > 1:
        log_x = np.log10(df_validos['TotalCases'])
        log_y = np.log10(df_validos['TotalDeaths'])
        coef = np.polyfit(log_x, log_y, 1)
        x_linea = np.logspace(np.log10(df['TotalCases'].min()), 
                               np.log10(df['TotalCases'].max()), 100)
        y_linea = 10**(coef[0] * np.log10(x_linea) + coef[1])
        
        fig.add_trace(go.Scatter(
            x=x_linea, y=y_linea, mode='lines', name='Línea de Tendencia',
            line=dict(color='rgba(150, 150, 150, 0.5)', width=2, dash='dash'),
            hoverinfo='skip', showlegend=True
        ))
    
    # LAYOUT
    fig.update_layout(
        title=dict(
            text='🫧 Análisis Multidimensional COVID-19<br>'
                 '<sub>Tamaño = Población | Color = Continente | Posición = Casos vs Muertes</sub>',
            x=0.5, xanchor='center',
            font=dict(size=26, color='#1e3a8a', family='Inter', weight=700)
        ),
        xaxis=dict(
            title='<b>Casos Confirmados Totales</b>', type='log',
            showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)',
            tickformat=',', title_font=dict(size=15, color='#1e40af')
        ),
        yaxis=dict(
            title='<b>Muertes Totales</b>', type='log',
            showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)',
            tickformat=',', title_font=dict(size=15, color='#dc2626')
        ),
        legend=dict(
            title=dict(text='<b>Continentes</b>', font=dict(size=14)),
            orientation="v", yanchor="top", y=0.99, xanchor="right", x=0.99,
            bgcolor='rgba(255, 255, 255, 0.9)', bordercolor='gray', borderwidth=2
        ),
        paper_bgcolor='rgba(240, 248, 255, 0.5)', plot_bgcolor='white',
        height=750, margin=dict(t=150, b=100, l=100, r=100),
        hovermode='closest'
    )
    
    graph_html = pio.to_html(fig, full_html=False, config={
        'displayModeBar': True, 'displaylogo': False,
        'modeBarButtonsToRemove': ['lasso2d', 'select2d']
    })
    
    return render_template('dashboard.html', graph_html=graph_html,
                         titulo="Análisis Multidimensional - Gráfico de Burbujas",
                         tipo_grafico="burbujas")


# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)


