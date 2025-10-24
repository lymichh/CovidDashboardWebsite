"""
Connects to a SQL database using pyodbc
"""
from os import getenv
from flask import Flask, jsonify, render_template
import pyodbc

app = Flask(__name__)

# --- CONFIGURACIÓN DE LA CONEXIÓN ---
# Es una mejor práctica usar variables de entorno para datos sensibles.
# Si no se definen, se usarán los valores predeterminados que tenías.
server = 'localhost,1433'
database = 'COVID_19'
username = 'project'
password = 'project123'

# Cadena de conexión
connection_string = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'UID={username};'
    f'PWD={password}'
)

def get_db_data():
    """
    Se conecta a la base de datos, ejecuta una consulta y devuelve los datos.
    Maneja la conexión y el cursor de forma segura.
    """
    db = None  # Inicializa la variable de conexión
    try:
        # Usar 'with' asegura que la conexión se cierre incluso si hay errores
        db = pyodbc.connect(connection_string)
        cursor = db.cursor()

        cursor.execute("SELECT * FROM covid_19_clean_complete")
        rows = cursor.fetchall()

        # Convierte los resultados a una lista de diccionarios para que sea formato JSON
        columns = [column[0] for column in cursor.description]
        data = [dict(zip(columns, row)) for row in rows]
        
        return data

    except pyodbc.Error as ex:
        # Captura errores específicos de pyodbc para un mejor diagnóstico
        sqlstate = ex.args[0]
        print(f"Error de base de datos (SQLSTATE: {sqlstate}): {ex}")
        return {"error": "No se pudo conectar o consultar la base de datos.", "details": str(ex)}
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
        return {"error": "Ocurrió un error inesperado en el servidor.", "details": str(e)}
    finally:
        # Asegúrate de que la conexión se cierre solo si se abrió con éxito
        if db:
            db.close()
            print("Conexión a la base de datos cerrada.")


@app.route('/info')
def info():
    return render_template('data.html')

@app.route('/data')
def get_data():
    """
    Endpoint de la API para obtener los datos de la base de datos.
    """
    data = get_db_data()
    # Si hay un error, devuelve una respuesta con un código de estado apropiado
    if "error" in data:
        return jsonify(data), 500
    return jsonify(data)

@app.route('/')
def home():
    """
    Ruta que sirve la página principal HTML.
    """
    return render_template('index.html')


if __name__ == '__main__':
    # El puerto se puede configurar también con variables de entorno
    port = int(getenv('PORT', 5000))
    app.run(debug=True, port=port)
