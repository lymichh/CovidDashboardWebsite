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

app = Flask(__name__)

# --- CONFIGURACIÓN DE LA CONEXIÓN ---

server = getenv('DB_SERVER', 'localhost\\SQLEXPRESS')
#database = getenv('DB_DATABASE', 'COVID-19')
#username = getenv('DB_USERNAME', 'project')
#password = getenv('DB_PASSWORD', 'project123')
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



#------- QUERY 1: EVOLUCION EN EL TIEMPO DE CASOS  EN UN PAIS

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
@app.route('/mapa2')
def mapa2():
    tipo = request.args.get('tipo', 'Confirmed')
    
    query = """
        SELECT  
            UID,
            Admin2 AS County,
            Province_State AS State,
            Country_Region,
            Lat,
            Long,
            Combined_Key,
            Date,
            Confirmed,
            Deaths
        FROM usa_county_wise
        WHERE Lat IS NOT NULL 
          AND Long IS NOT NULL
          AND Admin2 IS NOT NULL
          AND Country_Region = 'US'
          AND Confirmed > 0
    """
    
    df = run_query(query)
    
    print(f"📊 Registros iniciales: {len(df)}")
    
    # Convertir Date a datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Tomar solo el último registro por condado
    df_latest = df.sort_values('Date').groupby(['UID'], as_index=False).last()
    
    print(f"📍 Condados únicos: {len(df_latest)}")
    
    # Limpiar coordenadas
    df_latest['Lat'] = pd.to_numeric(df_latest['Lat'], errors='coerce')
    df_latest['Long'] = pd.to_numeric(df_latest['Long'], errors='coerce')
    df_latest['Confirmed'] = pd.to_numeric(df_latest['Confirmed'], errors='coerce').fillna(0)
    df_latest['Deaths'] = pd.to_numeric(df_latest['Deaths'], errors='coerce').fillna(0)
    
    # Detectar formato incorrecto de coordenadas
    if df_latest['Lat'].abs().max() > 90:
        print("⚠️ Corrigiendo formato de coordenadas")
        df_latest['Lat'] = df_latest['Lat'] / 1e6
        df_latest['Long'] = df_latest['Long'] / 1e6
    
    # Filtrar coordenadas válidas de USA (incluyendo Alaska y Hawaii)
    df_latest = df_latest[
        (df_latest['Lat'].between(18, 72)) &  # Incluye Hawaii (19°N) y Alaska (71°N)
        (df_latest['Long'].between(-180, -60)) &  # Incluye Alaska occidental
        (df_latest['Confirmed'] > 0)
    ]
    
    print(f"✅ Condados válidos: {len(df_latest)}")
    print(f"📐 Rango Lat: {df_latest['Lat'].min():.2f} a {df_latest['Lat'].max():.2f}")
    print(f"📐 Rango Long: {df_latest['Long'].min():.2f} a {df_latest['Long'].max():.2f}")
    
    # Calcular tasa de letalidad
    df_latest['Tasa_Letalidad'] = np.where(
        df_latest['Confirmed'] > 0,
        (df_latest['Deaths'] / df_latest['Confirmed'] * 100).round(2),
        0
    )
    
    # Generar figura
    fig = figure_map2(df_latest, tipo)
    graph_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False, div_id='mapa_graph')
    
    # Calcular estadísticas
    if df_latest.empty:
        stats = {
            'total_condados': 0,
            'total_casos': 0,
            'total_muertes': 0,
            'letalidad_promedio': 0,
            'condado_max': 'N/A',
            'casos_max': 0,
            'fecha': 'N/A'
        }
    else:
        max_row = df_latest.loc[df_latest['Confirmed'].idxmax()]
        
        stats = {
            'total_condados': len(df_latest),
            'total_casos': int(df_latest['Confirmed'].sum()),
            'total_muertes': int(df_latest['Deaths'].sum()),
            'letalidad_promedio': round((df_latest['Deaths'].sum() / df_latest['Confirmed'].sum() * 100), 2) if df_latest['Confirmed'].sum() > 0 else 0,
            'condado_max': f"{max_row['County']}, {max_row['State']}",
            'casos_max': int(max_row['Confirmed']),
            'fecha': str(df_latest['Date'].max().date())
        }
    
    print(f"📈 Stats: {stats}")
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'html': graph_html, 'stats': stats})
    
    return render_template('mapa2.html', graph_html=graph_html, tipo=tipo, stats=stats)


def figure_map2(df, tipo='Confirmed'):
    """Genera el mapa de USA por condados con burbujas."""
    
    config = {
        'Confirmed': {
            'label': 'Casos Confirmados',
            'color_scale': 'Reds',
            'column': 'Confirmed',
            'emoji': '😷'
        },
        'Deaths': {
            'label': 'Muertes',
            'color_scale': 'Oranges',
            'column': 'Deaths',
            'emoji': '💀'
        },
        'Letalidad': {
            'label': 'Tasa de Letalidad (%)',
            'color_scale': 'Blues',
            'column': 'Tasa_Letalidad',
            'emoji': '📊'
        }
    }
    
    selected_config = config.get(tipo, config['Confirmed'])
    column = selected_config['column']
    
    print(f"🎨 Creando mapa con columna: {column}")
    print(f"📊 Rango de valores: {df[column].min():.2f} a {df[column].max():.2f}")
    
    # Usar percentil 90 para mejor visualización de colores
    max_color = df[column].quantile(0.90)
    
    # Crear mapa de burbujas
    fig = px.scatter_mapbox(
        df,
        lat='Lat',
        lon='Long',
        size=column,
        color=column,
        hover_name='Combined_Key',
        hover_data={
            'County': True,
            'State': True,
            'Confirmed': ':,',
            'Deaths': ':,',
            'Tasa_Letalidad': ':.2f%',
            'Lat': False,
            'Long': False
        },
        color_continuous_scale=selected_config['color_scale'],
        range_color=(0, max_color),
        size_max=35,
        zoom=3.2,
        center={"lat": 37.5, "lon": -96},
        title=f'{selected_config["emoji"]} COVID-19 en USA por Condado: {selected_config["label"]}',
        labels={
            column: selected_config['label'],
            'County': 'Condado',
            'State': 'Estado',
            'Confirmed': 'Casos Confirmados',
            'Deaths': 'Muertes',
            'Tasa_Letalidad': 'Letalidad'
        }
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        height=700,
        margin=dict(l=0, r=0, t=60, b=0),
        title_x=0.5,
        title_font=dict(size=20, color="#1e40af", family="Arial, sans-serif", weight='bold'),
        paper_bgcolor="rgba(255,255,255,0.95)",
        coloraxis_colorbar=dict(
            title=selected_config['label'],
            thicknessmode="pixels",
            thickness=18,
            lenmode="pixels",
            len=350,
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.01,
            tickformat=',.0f' if tipo != 'Letalidad' else '.1f',
            ticksuffix='%' if tipo == 'Letalidad' else ''
        )
    )
    
    fig.update_traces(
        marker=dict(
            opacity=0.65,
            sizemode='diameter'
        )
    )
    
    return fig



#------- QUERY 2: MAPA DE CALOR
def figure_map(df, tipo='CasosPorMillon'):
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
        locations='Country_Region',
        featureidkey='properties.name',  # Coincide con los nombres del campo Country_Region
        color=tipo,
        color_continuous_scale= color_scale,
        range_color=(0, df[tipo].max()),
        mapbox_style="carto-positron",
        zoom=1,
        center={"lat": 20, "lon": 0},
        opacity=0.7,
        labels={tipo: label},
        title=f'Mapa de {label} por país'
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



# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')
    #return "<h1 style='text-align:center;'>Bienvenido al Dashboard COVID-19</h1><p style='text-align:center;'><a href='/evolucion'>Ir al Dashboard</a></p>"

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000)