from flask import Flask, render_template, jsonify, request
import plotly
import pyodbc
from os import getenv
import pandas as pd
import plotly.express as px
import plotly.io as pio
import json
import urllib.request
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

app = Flask(__name__)

# --- CONFIGURACIÓN DE LA CONEXIÓN ---

server = getenv('DB_SERVER', 'localhost\\SQLEXPRESS')

#server = 'localhost,1433'
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

# Función para ejecutar consultas SQL
def run_query(query, params=None):
    try:
        db = pyodbc.connect(connection_string)
        cursor = db.cursor()

        # Si hay parámetros, se pasan al ejecutar
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

#------- Pagina principal
@app.route("/home")
def homePage():
    return render_template("home.html")


#------- QUERY 1: EVOLUCION EN EL TIEMPO DE CASOS EN UN PAIS
@app.route('/evolucion')
def evolucion():
    """Ruta principal que maneja tanto la carga inicial como las actualizaciones."""
    
    # Obtener el país solicitado (default: Colombia)
    pais = request.args.get('pais', 'Colombia')
    
    db = pyodbc.connect(connection_string)
    cursor = db.cursor()

    # Obtener lista de países para el selector
    cursor.execute("SELECT DISTINCT Country_Region FROM dbo.covid_19_clean_complete ORDER BY Country_Region")
    paises = [row[0] for row in cursor.fetchall()]

    # Consulta de datos para el país seleccionado
    query = """
        SELECT Date, 
               SUM(Confirmed) AS TotalConfirmed, 
               SUM(Deaths) AS TotalDeaths, 
               SUM(Recovered) AS TotalRecovered
        FROM dbo.covid_19_clean_complete
        WHERE Country_Region = ?
        GROUP BY Date
        ORDER BY Date;
    """
    
    try:
        cursor.execute(query, pais)
        rows = cursor.fetchall()

        if not rows:
            print(f"⚠️ No se encontraron registros para: {pais}")
            db.close()
            return render_template('evolucion.html', 
                                 paises=paises, 
                                 pais_default=pais,
                                 graph_html='<p class="text-center text-red-500 font-semibold">No hay datos para este país.</p>')

        columns = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        
        print(f"✅ Datos cargados para {pais}: {len(df)} registros")

        # Generar la figura
        fig = figure_evolucion(df, pais)
        
        # Si es una solicitud AJAX (con header especial), devolver JSON
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            graph_json = pio.to_json(fig)
            return jsonify({'json': graph_json})
        
        # Si es carga inicial, devolver HTML completo
        graph_html = pio.to_html(fig, full_html=False, div_id='evolucion_graph')
        
        return render_template('evolucion.html', 
                             paises=paises, 
                             pais_default=pais,
                             graph_html=graph_html)

    except Exception as e:
        print(f"❌ Error al consultar datos: {e}")
        error_html = f'<p class="text-center text-red-500 font-semibold">Error al cargar datos: {str(e)}</p>'
        
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'html': error_html})

        return render_template('evolucion.html', 
                             paises=paises, 
                             pais_default=pais,
                             graph_html=error_html)
    
    finally:
        db.close()


def figure_evolucion(df, pais):
    """Genera la figura de Plotly para la evolución de casos."""
    fig = px.line(
        df, 
        x='Date', 
        y=df.columns[1:],  # Todas las columnas excepto Date
        labels={
            'value': 'Número de Casos',
            'variable': 'Categoría',
            'Date': 'Fecha'
        },
        title=f'{pais}',
        markers=True,
        line_shape='spline'
    )
    
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        font=dict(family="Inter, sans-serif", size=14, color="#333"),
        height=500,
        margin=dict(l=40, r=180, t=60, b=40),
        hovermode='x unified',
        legend=dict(
            title="Categoría",
            orientation="v",
            yanchor="top",
            y=1.02,
            xanchor="left",
            x=1.02
        )
    )
    
    # Mejorar colores de las líneas
    colors = {
        'TotalConfirmed': '#3498DB',
        'TotalDeaths': '#E74C3C',
        'TotalRecovered': '#2ECC71'
    }
    
    for trace in fig.data:
        if trace.name in colors:
            trace.line.color = colors[trace.name]
            trace.line.width = 3
    
    return fig

#------- FIN QUERY 1


#------- QUERY 2: MAPA DE CALOR
def figure_map(df, tipo='CasosPorMillon'):

    country_mapping = {
        'USA': 'United States of America',
        'UK': 'United Kingdom',
        'UAE': 'United Arab Emirates',
        'S. Korea': 'South Korea',
        'North Macedonia': 'Macedonia',
        'Congo': 'Republic of the Congo',
        'DRC': 'Democratic Republic of the Congo',
        'Czechia': 'Czech Republic',
        'CAR': 'Central African Republic',
        'Bosnia and Herzegovina': 'Bosnia and Herz.',
        'Dominican Republic': 'Dominican Rep.',
        'Equatorial Guinea': 'Eq. Guinea',
        'Timor-Leste': 'East Timor',
        "Côte d'Ivoire": 'Ivory Coast',
        'Eswatini': 'Swaziland',
        'Faeroe Islands': 'Faroe Islands',
        'Channel Islands': 'United Kingdom',
        'Saint Kitts and Nevis': 'St. Kitts and Nevis',
        'Saint Lucia': 'St. Lucia',
        'Saint Vincent and the Grenadines': 'St. Vincent and the Gren.',
        'Turks and Caicos': 'Turks and Caicos Is.',
        'British Virgin Islands': 'British Virgin Is.',
        'U.S. Virgin Islands': 'United States Virgin Islands',
        'Sint Maarten': 'Sint Maarten',
        'St. Barth': 'St-Barthélemy',
        'Caribbean Netherlands': 'Netherlands',
        'China': 'China'
    }

    df['Country_Mapped'] = df['Country_Region'].map(country_mapping).fillna(df['Country_Region'])



    # GeoJSON mundial (de Natural Earth, alojado en Plotly)
    df[tipo] = pd.to_numeric(df[tipo], errors='coerce')

    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    with urllib.request.urlopen(url) as response:
        geojson = json.load(response)

    
    label = "Casos por millón de habitantes" if tipo == "CasosPorMillon" else "Tasa de mortalidad (%)"
    color_scale = 'Reds' if tipo == "CasosPorMillon" else 'Blues'

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations='Country_Mapped',
        featureidkey='properties.name',  # Coincide con los nombres del campo Country_Region
        color=tipo,
        color_continuous_scale= color_scale,
        range_color=(0, df[tipo].quantile(0.98)),
        mapbox_style="carto-positron",
        zoom=1,
        center={"lat": 20, "lon": 0},
        opacity=0.8,
        labels={tipo: label},
        title=f'Mapa de {label} por país',
        hover_name='Country_Region',  
        hover_data={
            'Country_Mapped': False,  
            tipo: ':.2f'
        }
    )

    fig.update_traces(
    hoverlabel=dict(
        bgcolor="white",      # Fondo blanco
        font_color="black",   # Texto negro
        font_size=13,
        font_family="Inter, Arial",
        bordercolor="rgba(0,0,0,0.2)"  # Borde gris claro
    )
)

    
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        title_x=0.5,
        title_font=dict(size=22, color="#5850ec"),
        paper_bgcolor="rgba(0,0,0,0)",
        mapbox_accesstoken=None
    )
    return fig
    

@app.route('/mapa')
def mapa():
    tipo = request.args.get('tipo', 'CasosPorMillon')
    query = """
        SELECT 
                Country_Region,
                Population,
                (TotalCases * 1.0 / Population) * 1000000 AS CasosPorMillon,
                (TotalDeaths * 100.0 / TotalCases) AS TasaMortalidad
            FROM dbo.worldometer_data
            WHERE Population IS NOT NULL 
            AND Population > 0
            AND TotalCases > 0
            ORDER BY CasosPorMillon DESC;
    """
    
    df = run_query(query)
    fig = figure_map(df, tipo)
    graph_html = pio.to_html(fig, full_html=False, div_id='mapa_graph')

    return render_template('mapa.html', graph_html=graph_html, tipo=tipo)

#------- FIN QUERY 2


#------- Queery 3: Race-chart (top 15 paises con mas muertes en el tiempo)


@app.route("/race-chart")
def race_chart():
    query = """
    SELECT
        Date,
        Country_Region,
        WHO_Region,
        SUM(Confirmed) AS casos_totales,
        SUM(Deaths) AS muertes_totales
    FROM covid_19_clean_complete
    GROUP BY Date, Country_Region, WHO_Region
    HAVING SUM(Confirmed) > 0  -- Filtrar países sin casos
    ORDER BY Date, Country_Region;
    """
    df = run_query(query)
    
    print(f"Total de filas: {len(df)}")
    print(df.head())
    
    # Asegurar que Date es string en formato correcto
    df['Date'] = df['Date'].astype(str)
    
    # Filtrar solo fechas semanales (último día de la semana)
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df['week'] = df['Date_dt'].dt.to_period('W').astype(str)
    
    # Tomar el último día de cada semana por país
    df_weekly = df.sort_values('Date').groupby(['week', 'Country_Region', 'WHO_Region']).last().reset_index()
    
    
    # Para cada período, obtener el top 15
    top_n = 15
    df_top = df_weekly.groupby('week', group_keys=False).apply(
        lambda x: x.nlargest(top_n, 'muertes_totales')
    ).reset_index(drop=True)
    
    
    # Crear el gráfico
    fig = px.bar(
        df_top,
        x="muertes_totales",
        y="Country_Region",
        color="WHO_Region",
        animation_frame="week",  
        orientation='h',
        labels={
            "muertes_totales": "Muertes Totales",
            "Country_Region": "País",
            "WHO_Region": "Región OMS",
            "week": "Semana"
        },
        color_discrete_map={
            "Africa": "#E74C3C",
            "Americas": "#3498DB",
            "Eastern Mediterranean": "#F39C12",
            "Europe": "#9B59B6",
            "South-East Asia": "#1ABC9C",
            "Western Pacific": "#E67E22"
        },
        range_x=[0, df_top['muertes_totales'].max() * 1.1]  # Fijar rango X
    )
    
    # Mejorar la apariencia
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Muertes Totales",
        yaxis_title="",
        height=700,
        margin=dict(l=180, r=40, t=60, b=60),
        yaxis={
            'categoryorder': 'total ascending',
            'tickfont': dict(size=13, family="Arial")
        },
        xaxis={
            'tickformat': ',.0f',
            'gridcolor': '#E8E8E8'
        },
        font=dict(size=12, family="Arial"),
        showlegend=True,
        legend=dict(
            title=dict(text="Región OMS", font=dict(size=14, family="Arial Black")),
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#CCCCCC",
            borderwidth=1
        ),
        plot_bgcolor='#FAFAFA'
    )
    
    # Configurar animación (más fluida)
    fig.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 800  # milisegundos por frame
    fig.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 500
    
    # Configurar el slider
    fig.layout.sliders[0].pad = {"t": 50}
    fig.layout.sliders[0].currentvalue = {
        "prefix": "Semana: ",
        "font": {"size": 16, "color": "#1d4ed8"}
    }
    
    # Agregar valores en las barras
    fig.update_traces(
        texttemplate='<b>%{x:,.0f}</b>',
        textposition='outside',
        textfont=dict(size=12, family="Arial Black"),
        marker=dict(
            line=dict(width=0.5, color='white')
        )
    )
    
    graph_html = fig.to_html(include_plotlyjs='cdn', div_id='race-chart')
    return render_template("race_chart.html", graph_html=graph_html)

#------- FIN QUERY 3


#------- Query 4: KPIs principales

@app.route('/kpi')
def kpi_dashboard():
    # QUERY 1: Datos históricos (últimas 20 fechas)
    query_historico = """
        SELECT TOP 20
            Date,
            SUM(Confirmed) as TotalConfirmados,
            SUM(Deaths) as TotalMuertes,
            SUM(Recovered) as TotalRecuperados,
            SUM(Active) as TotalActivos
        FROM dbo.day_wise
        GROUP BY Date
        ORDER BY Date DESC
    """
    df_historico = run_query(query_historico)
    
    # QUERY 2: Estadísticas globales ACTUALES
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
    
    # QUERY 3: Tendencias semanales ACTUALES (para velocidad de propagación)
    query_tendencias = """
        SELECT 
            AVG(_1_week_increase) as PromedioAumentoSemanal
        FROM dbo.country_wise_latest
        WHERE _1_week_increase IS NOT NULL
    """
    df_tendencias = run_query(query_tendencias)
    
    # QUERY 4: Casos graves y testing ACTUALES
    query_testing = """
        SELECT 
            SUM(Serious_Critical) as CasosGraves,
            SUM(ActiveCases) as CasosActivosTotal,
            AVG(Tests_1M_pop) as PromedioTestsPorMillon
        FROM dbo.worldometer_data
        WHERE Tests_1M_pop IS NOT NULL
    """
    df_testing = run_query(query_testing)
    
    # CALCULAR KPIs ACTUALES (ESTÁTICOS - no cambian con el slider)
    if not df_global.empty:
        total_confirmados = df_global['TotalConfirmados'].iloc[0]
        total_muertes = df_global['TotalMuertes'].iloc[0]
        total_recuperados = df_global['TotalRecuperados'].iloc[0]
        paises_afectados = df_global['PaisesAfectados'].iloc[0]
        
        # KPIs que SÍ cambian con el tiempo (históricos)
        tasa_recuperacion_inicial = (total_recuperados / total_confirmados * 100) if total_confirmados > 0 else 0
        tasa_mortalidad_inicial = (total_muertes / total_confirmados * 100) if total_confirmados > 0 else 0
        
        # KPIs que NO cambian (estáticos, basados en datos actuales)
        velocidad_propagacion = abs(df_tendencias['PromedioAumentoSemanal'].iloc[0]) if not df_tendencias.empty else 0
        
        casos_graves = df_testing['CasosGraves'].iloc[0] if not df_testing.empty else 0
        casos_activos_total = df_testing['CasosActivosTotal'].iloc[0] if not df_testing.empty else 1
        carga_hospitalaria = (casos_graves / casos_activos_total * 100) if casos_activos_total > 0 else 0
        
        tests_por_millon = df_testing['PromedioTestsPorMillon'].iloc[0] if not df_testing.empty else 0
        
    else:
        tasa_recuperacion_inicial = tasa_mortalidad_inicial = velocidad_propagacion = 0
        carga_hospitalaria = paises_afectados = tests_por_millon = 0
        total_confirmados = total_muertes = total_recuperados = casos_graves = 0
    
    # CREAR FRAMES (solo para los 2 primeros gauges que cambian)
    frames = []
    slider_steps = []
    
    if not df_historico.empty:
        df_historico_sorted = df_historico.sort_values('Date')
        
        for idx, row in df_historico_sorted.iterrows():
            fecha = row['Date'].strftime('%d/%m/%Y')
            confirmados = row['TotalConfirmados']
            muertes = row['TotalMuertes']
            recuperados = row['TotalRecuperados']
            
            tasa_rec = (recuperados / confirmados * 100) if confirmados > 0 else 0
            tasa_mort = (muertes / confirmados * 100) if confirmados > 0 else 0
            
            # Frame solo actualiza los 2 primeros gauges (índices 0 y 1)
            frame = go.Frame(
                data=[
                    go.Indicator(value=tasa_rec),  # Gauge 1: Recuperación
                    go.Indicator(value=tasa_mort), # Gauge 2: Mortalidad
                ],
                name=fecha,
                traces=[0, 1]  # Solo actualiza los traces 0 y 1
            )
            frames.append(frame)
            
            slider_steps.append({
                'args': [[fecha], {'frame': {'duration': 300, 'redraw': True}, 'mode': 'immediate'}],
                'label': fecha,
                'method': 'animate'
            })
    
    # CREAR FIGURA CON 6 GAUGES (2x3 grid)
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
            [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]
        ],
        subplot_titles=(
            '🎯 Tasa de Recuperación (ANIMADO)',
            '☠️ Tasa de Mortalidad (ANIMADO)',
            '📈 Velocidad Semanal Propagación',
            '🏥 Carga Hospitalaria',
            '🌍 Países Afectados',
            '🧪 Tests por Millón'
        ),
        vertical_spacing=0.25,
        horizontal_spacing=0.12
    )
    
    # GAUGE 1: RECUPERACIÓN (ANIMADO)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=tasa_recuperacion_inicial,
        title={'text': f"<b>{total_recuperados:,.0f} recuperados</b>", 'font': {'size': 12}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#059669'}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%'},
            'bar': {'color': "#059669", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 50], 'color': '#fee2e2'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#10b981", 'width': 4},
                'thickness': 0.75,
                'value': tasa_recuperacion_inicial
            }
        }
    ), row=1, col=1)
    
    # GAUGE 2: MORTALIDAD (ANIMADO)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=tasa_mortalidad_inicial,
        title={'text': f"<b>{total_muertes:,.0f} fallecidos</b>", 'font': {'size': 12}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#dc2626'}},
        gauge={
            'axis': {'range': [0, 10], 'ticksuffix': '%'},
            'bar': {'color': "#dc2626", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 2], 'color': '#d1fae5'},
                {'range': [2, 5], 'color': '#fef3c7'},
                {'range': [5, 10], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 4},
                'thickness': 0.75,
                'value': tasa_mortalidad_inicial
            }
        }
    ), row=1, col=2)
    
    # GAUGE 3: VELOCIDAD (ESTÁTICO)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=velocidad_propagacion,
        title={'text': "<b>Promedio actual</b>", 'font': {'size': 12}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#f59e0b'}},
        gauge={
            'axis': {'range': [0, 50], 'ticksuffix': '%'},
            'bar': {'color': "#f59e0b", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 5], 'color': '#d1fae5'},
                {'range': [5, 15], 'color': '#fef3c7'},
                {'range': [15, 50], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#f97316", 'width': 4},
                'thickness': 0.75,
                'value': velocidad_propagacion
            }
        }
    ), row=1, col=3)
    
    # GAUGE 4: CARGA HOSPITALARIA (ESTÁTICO)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=carga_hospitalaria,
        title={'text': f"<b>{casos_graves:,.0f} casos graves</b>", 'font': {'size': 12}},
        number={'suffix': "%", 'font': {'size': 40, 'color': '#8b5cf6'}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': '%'},
            'bar': {'color': "#8b5cf6", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 30], 'color': '#d1fae5'},
                {'range': [30, 60], 'color': '#fef3c7'},
                {'range': [60, 100], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "#a855f7", 'width': 4},
                'thickness': 0.75,
                'value': carga_hospitalaria
            }
        }
    ), row=2, col=1)
    
    # GAUGE 5: PAÍSES (ESTÁTICO)
    fig.add_trace(go.Indicator(
        mode="number",
        value=paises_afectados,
        title={'text': "<b>Expansión global</b>", 'font': {'size': 12}},
        number={
            'font': {'size': 60, 'color': '#1e40af', 'family': 'Arial Black'},
            'suffix': ' países'
        },
        domain={'x': [0, 1], 'y': [0.2, 0.8]}
    ), row=2, col=2)
    
    # GAUGE 6: TESTS (ESTÁTICO)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=tests_por_millon,
        title={'text': "<b>Promedio mundial</b>", 'font': {'size': 12}},
        # Quitamos suffix con <br> y cualquier HTML problemático
        number={
            'font': {'size': 36, 'color': '#0891b2', 'family': 'Inter'},
            'valueformat': ',.0f'
        },
        gauge={
            'axis': {
                'range': [0, 300000],
                'tickformat': ',.0f',
                'tickvals': [0, 100000, 200000, 300000]
            },
            'bar': {'color': "#0891b2", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 100000], 'color': '#fee2e2'},
                {'range': [100000, 200000], 'color': '#fef3c7'},
                {'range': [200000, 300000], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "#06b6d4", 'width': 4},
                'thickness': 0.75,
                'value': tests_por_millon
            }
        },
        # Domain fijado para la celda (coincide con la tercera columna, fila 2)
        domain={'x': [0.66, 1.0], 'y': [0.00, 0.45]}
    ), row=2, col=3)

    # Annotation fija para la etiqueta "tests / 1M"
    fig.add_annotation(
        x=0.9,   # centrado aproximadamente sobre la tercera columna
        y=0.14,   # dentro de la segunda fila (ajusta +/- si quieres moverlo)
        xref='paper', yref='paper',
        text="<b>tests / 1M</b>",
        showarrow=False,
        font={'size': 12, 'color': '#0891b2', 'family': 'Inter'},
        align='center',
        bgcolor='rgba(255,255,255,0.0)'
    )
    
    fig.frames = frames
    fig.update_layout(
        autosize=True,
        paper_bgcolor='#f8fafc',
        height=750,
        margin=dict(t=100, b=100, l=60, r=60),
        font=dict(family='Inter, Arial', size=12),
        
        
        # Slider
        sliders=[{
            'y': 0, 'x': 0.15, 'len': 0.7,
            'currentvalue': {
                'prefix': '📅 Fecha: ',
                'font': {'size': 14, 'color': '#1e40af'}
            },
            'steps': slider_steps,
        }]
    )


    
    graph_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs='cdn',
        config={'responsive': True, 'displayModeBar': False}
    )


    
    return render_template('kpi.html',
                         graph_html=graph_html)


#------- FIN QUERY 4

#------- Query 5: Gráfico de pastel

@app.route('/pastel')
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

    # Paleta en tonos azules (de oscuro a claro)
    colores = ['#0D3B66', '#1E6091', '#2A6F97', '#468FAF', '#61A5C2', '#89C2D9']

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
            borderwidth=1,
            itemclick=False,
            itemdoubleclick=False
        ),
        annotations=[
            dict(
                text=f"<b style='font-size:18px;'>{total_mundial:,}</b><br>" +
                     f"<span style='font-size:12px;'>Muertes</span><br>" +
                     f"<span style='font-size:12px;'>totales</span>",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color='#065f46', family='Inter'),
                align='center',
                name='centerLabel'
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

    return render_template('pastel.html',
                        graph_html=graph_html)

#------ FIN QUERY 5

# Ruta principal
@app.route('/')
def home():
    return render_template('home.html')
    #return "<h1 style='text-align:center;'>Bienvenido al Dashboard COVID-19</h1><p style='text-align:center;'><a href='/evolucion'>Ir al Dashboard</a></p>"

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True, port=5050)