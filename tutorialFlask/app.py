from flask import Flask, render_template, jsonify
import pyodbc
import pandas as pd
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

# --- CONFIGURACIÓN DE LA CONEXIÓN ---
# Es una mejor práctica usar variables de entorno para datos sensibles.
# Si no se definen, se usarán los valores predeterminados que tenías.
#server = getenv('DB_SERVER', 'localhost\\SQLEXPRESS')
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

    fig = px.line(df, x='Date', y='TotalConfirmed',
                  title=f'Evolución de Casos Confirmados en {pais_default}',
                  markers=True,
                  line_shape='spline')
    fig.update_layout(
        template='plotly_white',
        title_x=0.5,
        font=dict(family="Inter, sans-serif", size=14, color="#333"),
        height=500,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    graph_html = pio.to_html(fig, full_html=False)
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
        fig = px.line(
            df,
            x='Date',
            y='TotalConfirmed',
            title=f'Evolución de casos confirmados en {pais}',
            markers=True
        )

        graph_html = pio.to_html(fig, full_html=False)
        return jsonify({'html': graph_html})

    except Exception as e:
        print("❌ Error al ejecutar consulta:", e)
        return jsonify({'html': f'<p style="color:red;">Error: {str(e)}</p>'})

    finally:
        if db:
            db.close()


@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)