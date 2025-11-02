from flask import Flask, render_template, jsonify
import pyodbc
from os import getenv
import pandas as pd
import plotly.express as px
import plotly.io as pio
import json
import urllib.request

app = Flask(__name__)

# --- CONFIGURACIÓN DE LA CONEXIÓN ---

##server = getenv('DB_SERVER', 'localhost\\SQLEXPRESS')
#database = getenv('DB_DATABASE', 'COVID-19')
#username = getenv('DB_USERNAME', 'project')
#password = getenv('DB_PASSWORD', 'project123')
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
def figure_map(df):
    # GeoJSON mundial (de Natural Earth, alojado en Plotly)
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"
    with urllib.request.urlopen(url) as response:
        geojson = json.load(response)

    fig = px.choropleth_mapbox(
        df,
        geojson=geojson,
        locations='Country_Region',
        featureidkey='properties.name',  # Coincide con los nombres del campo Country_Region
        color='TotalConfirmed',
        color_continuous_scale="Reds",
        range_color=(0, df['TotalConfirmed'].max()),
        mapbox_style="carto-positron",
        zoom=1,
        center={"lat": 20, "lon": 0},
        opacity=0.7,
        labels={'TotalConfirmed': 'Casos Confirmados'},
        title='Mapa de Casos Confirmados por País'
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
    query = """
        SELECT Country_Region, 
               SUM(Confirmed) AS TotalConfirmed
        FROM dbo.covid_19_clean_complete
        GROUP BY Country_Region;
    """
    
    df = run_query(query)
    fig = figure_map(df)
    graph_html = pio.to_html(fig, full_html=False, div_id='mapa_graph')

    return render_template('mapa.html', graph_html=graph_html)

#------- FIN QUERY 2

#------- QUERY 3: COMPARACION PAISES VS CASOS CONFIRMADOS
@app.route('/comparacion')
def comparacion():
    query = """
        SELECT TOP 10 Country_Region, SUM(Confirmed) AS TotalConfirmados
        FROM dbo.covid_19_clean_complete
        GROUP BY Country_Region
        ORDER BY TotalConfirmados DESC;
    """
    df = run_query(query)

    fig = px.bar(
    df,
    x='TotalConfirmados',
    y='Country_Region',
    orientation='h',
    color_discrete_sequence=['#7c3aed'],  # 💜 morado vibrante
    text='TotalConfirmados',
    title='Top 10 países con más casos confirmados'
    )

    # Ajustes visuales para barras sólidas y uniformes
    fig.update_traces(
        marker=dict(
            color='#7c3aed',      
            line=dict(color='#4c1d95', width=1.5)  
        ),
        opacity=1,                
        texttemplate='%{text:,}', 
        textposition='inside',     
        insidetextanchor='start',
        insidetextfont=dict(color='white', size=13)
    )

    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Casos: %{x:,}',
        texttemplate='%{text:,}',  # separa miles con coma
        textposition='outside'
    )

    # Convertimos el gráfico a HTML embebible
    import plotly.io as pio
    graph_html = pio.to_html(fig, full_html=False)

    return render_template('dashboard.html', graph_html=graph_html, titulo="Comparación Global de Casos COVID-19")

#------- FIN QUERY 3

#------- QUERY 4: DISTRIBUCIÓN DE MUERTES POR CONTINENTE

@app.route('/muerte_continente')
def muerte_continente():
    query = """
        SELECT 
            w.Continent as Continente, SUM(c.Deaths) AS TotalMuertes
        FROM dbo.country_wise_latest c
        JOIN dbo.worldometer_data w
            ON c.Country_Region = w.Country_Region
        WHERE w.Continent IS NOT NULL
        GROUP BY w.Continent
        ORDER BY TotalMuertes DESC;
    """
    df = run_query(query)

    fig = px.pie(
        df,
        names='Continente',
        values='TotalMuertes',
        title='Distribución de muertes por continente',
        color_discrete_sequence=px.colors.sequential.Greens_r,
        hole=0.4 
    )   

    fig.update_traces(
        textinfo='label+percent',
        textposition='outside',
        insidetextorientation='radial',
        textfont=dict(size=14, color='#333'),
        hovertemplate='<b>%{label}</b><br>Muertes: %{value:,}<br>Porcentaje: %{percent}',
        pull=[0.05 if i == 0 else 0 for i in range(len(df))],     # resalta el más alto
        marker=dict(line=dict(color='white', width=2))
    )

    fig.update_layout(
        title=dict(
            text='Distribución de muertes por continente',
            x=0.5,  # centrado
            font=dict(size=24, color='#4c1d95', family='Inter')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=80, b=80, l=100, r=100),  # más aire alrededor del gráfico
        legend=dict(
            orientation="h",  # horizontal
            y=-0.2,
            x=0.5,
            xanchor='center',
            yanchor='top',
            font=dict(size=13)
        )
    )
    
    graph_html = pio.to_html(fig, full_html=False)

    return render_template('dashboard.html', graph_html=graph_html, titulo="Distribución de muertes por continente")

#------- FIN QUERY 3

# Ruta principal
@app.route('/')
def home():
    return render_template('index.html')
    #return "<h1 style='text-align:center;'>Bienvenido al Dashboard COVID-19</h1><p style='text-align:center;'><a href='/evolucion'>Ir al Dashboard</a></p>"


if __name__ == '__main__':
    app.run(debug=True, port=5000)