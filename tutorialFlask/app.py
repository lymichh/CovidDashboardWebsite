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

def figure_evolucion(df, pais):
    fig = px.line(df, x='Date', y=df.columns[1:], labels={
        'value': 'Número de Casos',
        'variable': 'Categoría',
        'Fecha': 'Fecha'},
                  title=f'Evolución de Casos Confirmados en {pais}',
                  markers=True,
                  line_shape='spline')
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        font=dict(family="Inter, sans-serif", size=14, color="#333"),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig


@app.route('/evolucion')
def evolucion():

    db = pyodbc.connect(connection_string)
    cursor = db.cursor()

    cursor.execute("SELECT DISTINCT Country_Region FROM dbo.covid_19_clean_complete ORDER BY Country_Region")
    paises = [row[0] for row in cursor.fetchall()]

    pais_default = 'Colombia'

    """Genera la gráfica de evolución de casos confirmados en Colombia."""
    query = """
        SELECT Date, 
               SUM(Confirmed) AS TotalConfirmed, 
               SUM(Deaths) AS TotalMuertes, 
               SUM(Recovered) AS TotalRecuperados
        FROM dbo.covid_19_clean_complete
        WHERE Country_Region = ?
        GROUP BY Date
        ORDER BY Date;
    """
    cursor.execute(query, pais_default)
    rows = cursor.fetchall()

    columns = [column[0] for column in cursor.description]
    df = pd.DataFrame.from_records(rows, columns=columns)

    fig = figure_evolucion(df, pais_default)

    graph_html = pio.to_html(fig, full_html=False, div_id='evolucion_graph')
    db.close()
    return render_template('dashboard.html', paises=paises, pais_default = pais_default, graph_html=graph_html)

@app.route('/evolucion_data/<pais>')
def evolucion_data(pais):
    print(f"🔍 Consultando datos para: {pais}")  # Para verificar en consola

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
        db = pyodbc.connect(connection_string)
        cursor = db.cursor()
        cursor.execute(query, pais)
        rows = cursor.fetchall()

        if not rows:
            print("⚠️ No se encontraron registros.")
            return jsonify({'html': '<p class="text-center text-red-500 font-semibold">No hay datos para este país.</p>'})

        columns = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(rows, columns=columns)
        print(df.head())  # Para ver si sí trajo datos

        # Gráfico simple por ahora (solo TotalConfirmed)
        fig = figure_evolucion(df, pais)

        graph_json = pio.to_json(fig)
        return jsonify({'json': graph_json})

    except Exception as e:
        print("❌ Error al ejecutar consulta:", e)
        return jsonify({'html': f'<p style="color:red;">Error: {str(e)}</p>'})

    finally:
        if db:
            db.close()

#------- FIN QUERY 1


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
        title=f'Mapa de {label} por País'
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        title_x=0.5,
        title_font=dict(size=22, color="#1e3a8a"),
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
    
    # Limpiar datos
    #df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    #df["casos_totales"] = df["casos_totales"].clip(lower=0)
    #df["muertes_totales"] = df["muertes_totales"].clip(lower=0)
    
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
        title="🏆 Top 15 Países con más Muertes de COVID-19",
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
        title_font=dict(size=24, color="#1d4ed8", family="Arial Black"),
        height=700,
        margin=dict(l=180, r=40, t=100, b=60),
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