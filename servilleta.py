import streamlit as st
import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components
from PIL import Image
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import osmnx as ox
from geopy.distance import geodesic
import streamlit.components.v1 as components
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import graphviz

# Configuración de la aplicación
st.set_page_config(page_title="Visualización de Marcos SCVID", layout="wide")
# Display the image above the title
st.image('Napkin App.png', use_column_width=True)

def add_background_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        .block-container {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 50px;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Ruta de la imagen local
image_path = "Servilleta App.jpg"  # Reemplaza con la ruta a tu imagen local

# Aplicar la imagen de fondo
add_background_local(image_path)

# Agregar una nueva pestaña para la reseña del libro
tabs = st.tabs([ "Reseña del Libro", "Quién/Qué", "Cuánto", "Dónde", "Cuándo", "Cómo", "Por qué","Acerca de mí"])

# Pestaña: Reseña del libro
with tabs[0]:

    resumen = """
    ## Resumen del libro: "The Back of the Napkin de Dan Roam"
    
    Este resumen se centra en el marco de 6 preguntas, el SQVID y las estrategias visuales recomendadas en el libro 
    "The Back of the Napkin" de Dan Roam para resolver problemas y vender ideas.
    
    El libro argumenta que el pensamiento visual, utilizando dibujos sencillos, es una herramienta poderosa para la 
    resolución de problemas y la comunicación efectiva. Roam explica que cualquier problema se puede aclarar con una 
    imagen, y cualquier imagen se puede crear utilizando un conjunto simple de herramientas y reglas.
    
    ### El Marco de las 6 Preguntas (6W)
    
    Roam propone un marco de seis preguntas fundamentales, conocidas como las 6W, para analizar y abordar cualquier problema:  
    
    * **Quién/Qué (Who/What):** Define los actores y objetos involucrados.
    * **Cuánto (How Much):**  Analiza las cantidades, medidas y datos.
    * **Dónde (Where):**  Ubica el problema en un contexto espacial.
    * **Cuándo (When):**  Establece un marco temporal para el problema.
    * **Cómo (How):**  Describe los procesos y relaciones de causa y efecto.
    * **Por qué (Why):**   Explora las razones, motivaciones y causas subyacentes. 
    
    Estas preguntas no solo ayudan a comprender el problema, sino que también guían la elección del gráfico o estrategia 
    visual más efectiva para comunicarlo.
    
    ### El SQVID como Herramienta de Imaginación
    
    El SQVID es un acrónimo que representa cinco preguntas para activar la imaginación visual y explorar diferentes 
    perspectivas de una idea:
    
    * **Simple vs. Elaborado:** ¿Se necesita una representación simple o detallada?
    * **Cualitativo vs. Cuantitativo:** ¿Se enfatizan las características o los datos numéricos?
    * **Visión vs. Ejecución:** ¿Se busca inspirar o mostrar los pasos concretos?
    * **Individual vs. Comparación:** ¿Se presenta una sola idea o se compara con otras?
    * **Cambio vs. Status Quo:** ¿Se propone una transformación o se describe la situación actual?
    
    Al responder estas preguntas, se pueden generar múltiples representaciones visuales de una idea y elegir la más 
    efectiva para la audiencia y el objetivo.
    
    ### Gráficos y Estrategias Visuales Recomendadas
    
    El libro presenta seis marcos visuales principales, cada uno correspondiente a una de las 6W:
    
    * **Retrato:** Para mostrar **quién/qué** (ej. un organigrama, un diagrama de personajes).
    * **Gráfico:** Para visualizar **cuánto** (ej. gráfico de barras, gráfico circular).
    * **Mapa:** Para representar **dónde** (ej. mapa mental, diagrama de flujo de procesos).
    * **Línea de Tiempo:** Para ilustrar **cuándo** (ej. cronograma, diagrama de Gantt).
    * **Diagrama de Flujo:** Para explicar **cómo** (ej. diagrama de flujo de trabajo, algoritmo).
    * **Gráfico de Múltiples Variables:** Para analizar **por qué** (ej. gráfico de dispersión, mapa de calor).
    
    En resumen, "The Back of the Napkin" ofrece un enfoque práctico para utilizar el pensamiento visual como herramienta 
    para la resolución de problemas y la comunicación efectiva.
    """

    st.markdown(resumen)
    
    st.subheader("Razones de la escogencia de los gráficos")
    
    # Ejemplo de razones
    explicacion_graficos = """
    Los gráficos utilizados en esta aplicación se alinean con el marco visual propuesto por Dan Roam en "The Back of the Napkin". 
    
    1. **Quién/Qué (Gráfico de Nodos):** Este gráfico permite mostrar las conexiones entre personas, estudios y habilidades, siguiendo la recomendación de usar retratos o gráficos de red para visualizar relaciones entre actores y objetos.
    
    2. **Cuánto (Gráfico de Pareto):** Este gráfico es ideal para visualizar cantidades y datos. En particular, resalta la importancia de unos pocos elementos clave sobre la mayoría, una estrategia visual efectiva para mostrar patrones cuantitativos.
    
    3. **Cómo (Diagrama de Flujo):** El uso de diagramas de flujo para mostrar procesos es una herramienta clara y directa para entender relaciones de causa y efecto.
    
    4. **Cuándo (Línea de Tiempo):** Para mostrar secuencias temporales, la línea de tiempo es el gráfico recomendado para organizar eventos de manera clara y ordenada.
    
    5. **Por qué (Gráfico de Burbujas):** El gráfico de burbujas permite analizar múltiples variables simultáneamente, destacando cómo las categorías se comparan en función de varias dimensiones (eje X, eje Y, tamaño y color de las burbujas). Esta visualización es útil para explorar las razones subyacentes y patrones complejos en los datos, facilitando la comprensión de cómo diferentes variables se relacionan entre sí.
    """
    st.markdown(explicacion_graficos) 

    # Explicación en audio precargada
    st.subheader("AI Podcast del libro (EN)")
    
    # Ruta del archivo de audio precargado
    audio_path = "Napkin_podcast.mp3"  # Cambia esta ruta al archivo correcto
    audio_file = open(audio_path, "rb").read()  # Cargar el archivo de audio

    # Reproducir el archivo de audio
    st.audio(audio_file, format="audio/mp3")

# Sidebar
st.sidebar.title("Datos de Entrada")
# Función para cargar imagen y convertirla en base64
def get_image_base64(image):
    if isinstance(image, str):  # Si la imagen es una ruta
        with open(image, "rb") as image_file:
            return "data:image/png;base64," + base64.b64encode(image_file.read()).decode("utf-8")
    elif isinstance(image, Image.Image):  # Si la imagen es un objeto PIL
        buffered = io.BytesIO()
        image = image.resize((150, 150))  # Ajustar tamaño de la imagen a 150x150 píxeles
        image.save(buffered, format="PNG")  # Convertir imagen a bytes
        return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    return None
    

def crear_grafico_quien_que(nombre, categorias, imagen):
    # Crear un network graph
    G = nx.Graph()

    # Agregar nodo principal (la persona o entidad)
    G.add_node(nombre, type='central', shape='circularImage', image=get_image_base64(imagen) if imagen else None)

    # Agregar nodos para las categorías y atributos
    for categoria, atributos in categorias.items():
        for atributo in atributos:
            G.add_node(atributo, type=categoria)
            G.add_edge(nombre, atributo)  # Conectar atributo con el nodo central

            # Conectar atributos entre sí si están en la misma categoría
            for otro_atributo in atributos:
                if atributo != otro_atributo:
                    G.add_edge(atributo, otro_atributo)

    # Crear visualización con pyvis
    person_net = Network(
        height='600px',
        width='100%',
        bgcolor='#222222',
        font_color='white'
    )

    # Definir colores para las categorías
    categoria_colores = {
        'Categoría 1': '#1f77b4',  # Azul oscuro
        'Categoría 2': '#2ca02c',  # Verde oscuro
        'Categoría 3': '#9467bd',  # Púrpura oscuro
        'Categoría 4': '#bcbd22',  # Amarillo oliva oscuro
    }

    # Configurar los nodos con imágenes, colores y tamaños
    for node in G.nodes(data=True):
        node_categoria = node[1].get('type', 'central')
        color = categoria_colores.get(node_categoria, 'gray')  # Color por defecto si no se encuentra la categoría

        # Determinar si el nodo es el central o uno de los otros
        if node_categoria == 'central':
            node_options = {
                "label": node[0],
                "shape": "circularImage" if node[1].get('image', '') else "circle",
                "image": node[1].get('image', ''),  # Convertir a base64 si es una imagen
                "color": color,
                "size": 80,  # Tamaño más grande para el nodo central
                "fixed": {"x": False, "y": False}  # Mantener el nodo fijo en tamaño, no en posición
            }
        else:
            node_options = {
                "label": node[0],
                "shape": "circle",
                "color": color,
                "size": 10,  # Tamaño más pequeño para los nodos secundarios
                "fixed": {"x": False, "y": False}  # Mantener el nodo fijo en tamaño, no en posición
            }
        person_net.add_node(node[0], **node_options)

    # Agregar edges
    for edge in G.edges():
        person_net.add_edge(edge[0], edge[1])

    # Configurar layout del grafo con centrado
    person_net.repulsion(
        node_distance=200,
        central_gravity=0.33,
        spring_length=100,
        spring_strength=0.10,
        damping=0.95
    )
    
    # Agregar opciones para mantener el gráfico centrado
    person_net.set_options("""
        var options = {
            "physics": {
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000
                },
                "minVelocity": 0.75
            },
            "interaction": {
                "dragNodes": true,
                "zoomView": true
            }
        }
    """)

    # Guardar y mostrar grafo en HTML
    path = '/tmp'
    person_net.save_graph(f'{path}/pyvis_graph.html')
    
    with open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8') as HtmlFile:
        graph_html = HtmlFile.read()

    # Mostrar grafo en la app con Streamlit Components con ancho responsivo
    components.html(graph_html, height=600, width=800)

# Pestaña: Quién/Qué
with tabs[1]:
    st.header("¿Quién/Qué?")
    st.sidebar.subheader("Ingresos de datos del ¿Quién/Qué?:")

    # Entrada de texto para el nombre
    nombre = st.sidebar.text_input("Ingrese el nombre:", "Ai-ngineering")

    # Imagen predeterminada si no se carga ninguna
    imagen_predeterminada = "perfil.jpg"

    # Cargar una imagen
    imagen_subida = st.sidebar.file_uploader("Cargue una foto", type=["png", "jpg", "jpeg"])
    imagen = Image.open(imagen_subida) if imagen_subida else imagen_predeterminada

    # Ingreso de categorías y atributos
    st.sidebar.write("Ingrese las diferentes categorías y atributos:")

    # Crear DataFrame editable para las categorías
    example_data = {
        "Categoría 1": ["Ingeniero", "Big Data", "MBA"],
        "Categoría 2": ["Procesos", "Razonamiento", "Cálculo", "Datos", "Storytelling", "Programación"],
        "Categoría 3": ["Esposo", "Padre", "Hijo", "Músico"],
        "Categoría 4": ["Venezuela", "Colombia", "Ecuador", "México", "Brasil"]
    }
    df_categorias = pd.DataFrame.from_dict(example_data, orient='index').transpose()
    
    # Data editor en el sidebar
    df_categorias = st.sidebar.data_editor(df_categorias, num_rows="dynamic", key="df_quienque")

    # Convertir el DataFrame a un diccionario de listas para categorías
    categorias = {col: df_categorias[col].dropna().tolist() for col in df_categorias.columns}

    # Mostrar el gráfico solo si se ha ingresado un nombre
    if nombre:
        crear_grafico_quien_que(nombre, categorias, imagen)
            # Agregar un enlace al final de la pestaña
    st.markdown("Para más información consulta en: [Interactive network visualizations](https://pyvis.readthedocs.io/en/latest/)")

# Función para crear gráfico de Pareto
def crear_grafico_pareto(datos):
    # Ordenar los datos de mayor a menor
    datos = datos.sort_values(by='Valor', ascending=False).reset_index(drop=True)
    
    # Calcular el valor acumulado y su porcentaje
    datos['Acumulado'] = datos['Valor'].cumsum()
    total = datos['Valor'].sum()
    datos['% Valor'] = datos['Valor'] / total * 100
    datos['% Acumulado'] = datos['Acumulado'] / total * 100
    
    # Crear gráfico de Pareto
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Crear colormap para las barras
    norm = mcolors.Normalize(vmin=datos['Valor'].min(), vmax=datos['Valor'].max())
    cmap = plt.get_cmap('inferno')  # Puedes cambiar 'inferno' por otro colormap
    
    # Ajustar el ancho de las barras para que no haya separación
    ancho_barras = 1.0
    
    # Gráfico de barras con colores de heatmap
    bars = ax1.bar(datos['Categoría'], datos['% Valor'], color=cmap(norm(datos['Valor'])), width=ancho_barras)
    ax1.set_ylabel('Porcentaje del Valor')
    ax1.set_xlabel('Categoría')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 100)  # Configurar el rango del eje y
    
    # Crear un segundo eje para la línea acumulada
    ax2 = ax1.twinx()
    ax2.plot(datos['Categoría'], datos['% Acumulado'], color='red', marker='D', linestyle='-', label='Porcentaje Acumulado')
    ax2.axhline(80, color='gray', linestyle='--')  # Línea que marca el 80%
    ax2.set_ylabel('Porcentaje Acumulado')
    ax2.set_ylim(0, 100)  # Configurar el rango del eje y
    
    # Añadir leyenda y etiquetas
    plt.title('Gráfico de Pareto')
    ax2.legend(loc='best')
    
    # Añadir barra de color
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, orientation='vertical')
    cbar.set_label('Valor')
    
    st.pyplot(fig)
    
# Pestaña: Cuánto (modificada)
with tabs[2]:
    st.header("¿Cuánto?")
    # Ejemplo de datos en DataFrame (similar al formato que mencionaste)
    st.sidebar.subheader("Ingresos de datos del ¿Cuánto?:")    
    
    example_data = {
        "Categoría": ["A", "B", "C", "D", "E", "F"],
        "Valor": [10, 40, 20, 15, 5, 2]
    }
    example_df = pd.DataFrame(example_data)
    
    # Mostrar ejemplo de datos en el sidebar
    st.sidebar.write("Ingrese las categorías y valores en el siguiente formato:")
    
    # Permitir al usuario editar el DataFrame
    df_cuanto = st.sidebar.data_editor(example_df, num_rows="dynamic", key="df_cuanto")

    # Verificar si el DataFrame tiene datos antes de generar el gráfico
    if not df_cuanto.empty:
        # Llamada a la función para crear el gráfico de Pareto
        crear_grafico_pareto(df_cuanto)
    st.markdown("Para más información consulta en: [Pareto chart](https://asq.org/quality-resources/pareto)")
        
# Función para obtener las coordenadas de un lugar utilizando OSMNX
def obtener_coordenadas_lugar(lugar):
    return ox.geocode(lugar)

# Función para calcular la distancia entre dos puntos (coordenadas)
def calcular_distancia(origen, destino):
    return geodesic(origen, destino).kilometers

# Función para graficar las ubicaciones en un mapa y mostrar la distancia, conectando los puntos con una línea
def crear_grafico_lugares(origen, destino, coordenadas_origen, coordenadas_destino):
    # Calcular la distancia entre las dos ubicaciones
    distancia = calcular_distancia(coordenadas_origen, coordenadas_destino)

    # Crear un DataFrame para los dos puntos
    df = pd.DataFrame({
        "Lugar": [origen, destino],
        "Latitud": [coordenadas_origen[0], coordenadas_destino[0]],
        "Longitud": [coordenadas_origen[1], coordenadas_destino[1]]
    })

    # Crear el mapa con los dos puntos
    fig = px.scatter_mapbox(
        df, lat="Latitud", lon="Longitud", zoom=5, height=600, text="Lugar"
    )

    # Añadir la línea que conecta los dos puntos
    fig.add_scattermapbox(
        lat=[coordenadas_origen[0], coordenadas_destino[0]],  # Latitudes de origen y destino
        lon=[coordenadas_origen[1], coordenadas_destino[1]],  # Longitudes de origen y destino
        mode="lines",
        line=dict(width=2, color="blue"),
        name="Línea de conexión"
    )

    fig.update_layout(mapbox_style="open-street-map")

    # Mostrar el gráfico en la aplicación
    st.plotly_chart(fig)

    # Mostrar la distancia entre los dos lugares
    st.write(f"La distancia entre {origen} y {destino} es de aproximadamente {distancia:.2f} km.")

# Pestaña: Dónde (modificada)
with tabs[3]:
    st.header("¿Dónde?")
    # Ejemplo de datos en DataFrame (similar al formato que mencionaste)
    st.sidebar.subheader("Ingresos de datos del ¿Dónde?:")    
    # Ingreso de las ubicaciones de origen y destino
    origen = st.sidebar.text_input("Ingrese el lugar de origen:", "Valencia, Venezuela")
    destino = st.sidebar.text_input("Ingrese el lugar de destino:", "Medellín, Colombia")

    # Mostrar el gráfico de los puntos y la distancia si se ingresan ambas ubicaciones
    if origen and destino:
        try:
            coordenadas_origen = obtener_coordenadas_lugar(origen)
            coordenadas_destino = obtener_coordenadas_lugar(destino)
            
            crear_grafico_lugares(origen, destino, coordenadas_origen, coordenadas_destino)
        except Exception as e:
            st.error(f"Error al obtener las coordenadas: {e}")

# Función para crear gráfico de Gantt usando Plotly Timeline
def crear_grafico_gantt(eventos):
    # Crear el gráfico de Gantt con Plotly
    fig = px.timeline(
        eventos,
        x_start="Fecha de Inicio", 
        x_end="Fecha de Fin", 
        y="Evento", 
        color='Categoría',  # Categorías para colorear las tareas
        title='Gráfico de Gantt de Eventos'
    )

    # Añadir el porcentaje de realización como etiquetas en el gráfico
    for i, row in eventos.iterrows():
        fig.add_annotation(
            x=row['Fecha de Fin'], 
            y=row['Evento'],
            text=f"{row['Porcentaje de Realización']}% completado", 
            showarrow=False,
            font=dict(size=12, color='white'),
            bgcolor="green" if row['Porcentaje de Realización'] == 100 else "blue",
            bordercolor="black",
            xanchor='left'
        )

    fig.update_yaxes(autorange="reversed")  # Invertir el eje Y
    fig.update_layout(height=600, width=900)  # Ajustar tamaño del gráfico
    st.plotly_chart(fig)

with tabs[4]:
    st.header("¿Cuándo?")

    # Ejemplo de datos ampliados en DataFrame
    st.sidebar.subheader("Ingresos de datos del ¿Cuándo?:")
    example_data = {
        "Evento": ["Proyecto Inicio", "Análisis de Requisitos", "Desarrollo", "Pruebas", "Implementación", "Cierre"],
        "Fecha de Inicio": ["2023-01-01", "2023-02-01", "2023-06-01", "2023-08-01", "2023-10-01", "2023-11-01"],
        "Fecha de Fin": ["2023-02-01", "2023-06-01", "2023-08-01", "2023-10-01", "2023-11-01", "2023-12-01"],
        "Categoría": ["Planificación", "Análisis", "Desarrollo", "Pruebas", "Implementación", "Cierre"],
        "Porcentaje de Realización": [100, 100, 80, 50, 20, 0]  # Porcentajes de realización
    }
    example_df = pd.DataFrame(example_data)

    # Ingreso de datos de tareas mediante un DataFrame editable
    st.sidebar.write("Ingrese los datos de las tareas en el siguiente formato:")
    df_tareas = st.sidebar.data_editor(example_df, num_rows="dynamic", key="df_gantt")

    # Verificar que el DataFrame tiene datos para generar el gráfico
    if not df_tareas.empty:
        # Convertir las columnas de fechas a formato de fecha
        df_tareas['Fecha de Inicio'] = pd.to_datetime(df_tareas['Fecha de Inicio'])
        df_tareas['Fecha de Fin'] = pd.to_datetime(df_tareas['Fecha de Fin'])

        # Crear gráfico Gantt a partir del DataFrame ingresado
        crear_grafico_gantt(df_tareas)
    
    st.markdown("Para más información consulta en: [Gantt chart](https://www.apm.org.uk/resources/find-a-resource/gantt-chart/)")


# # Función para crear gráfico de Gantt usando Plotly Timeline
# def crear_grafico_gantt(eventos):
#     fig = px.timeline(
#         eventos,
#         x_start="Fecha de Inicio", 
#         x_end="Fecha de Fin", 
#         y="Evento", 
#         color='Categoría',  # Esto es opcional si deseas agregar categorías para colorear las tareas
#         title='Gráfico de Gantt de Eventos'
#     )

#     fig.update_yaxes(autorange="reversed")  # Invertir el eje y para que las tareas se vean de arriba hacia abajo
#     fig.update_layout(height=600, width=900)  # Ajustar el tamaño del gráfico
#     st.plotly_chart(fig)

# with tabs[4]:
#     st.header("¿Cuándo?")

#     # Ejemplo de datos en DataFrame (similar al formato que mencionaste)
#     st.sidebar.subheader("Ingresos de datos del ¿Cuándo?:")
#     example_data = {
#         "Evento": ["Proyecto Inicio", "Desarrollo", "Fin"],
#         "Fecha de Inicio": ["2023-01-01", "2023-06-01", "2023-09-01"],
#         "Fecha de Fin": ["2023-06-01", "2023-09-01", "2023-12-01"],
#         "Categoría": ["Planificación", "Ejecución", "Cierre"]
#     }
#     example_df = pd.DataFrame(example_data)
#     #st.sidebar.write(example_df)

#     # Ingreso de datos de tareas mediante un DataFrame editable
#     st.sidebar.write("Ingrese los datos de las tareas en el siguiente formato:")
#     df_tareas = st.sidebar.data_editor(example_df, num_rows="dynamic", key="df_gantt")

#     # Verificar que el DataFrame tiene datos para generar el gráfico
#     if not df_tareas.empty:
#         # Convertir las columnas de fechas a formato de fecha
#         df_tareas['Fecha de Inicio'] = pd.to_datetime(df_tareas['Fecha de Inicio'])
#         df_tareas['Fecha de Fin'] = pd.to_datetime(df_tareas['Fecha de Fin'])

#         # Crear gráfico Gantt a partir del DataFrame ingresado
#         crear_grafico_gantt(df_tareas)
#     st.markdown("Para más información consulta en: [Gantt chart](https://www.apm.org.uk/resources/find-a-resource/gantt-chart/)")
    
# Pestaña: Cómo (modificada con el diagrama de flujo con simbología ANSI)
with tabs[5]:
    st.header("¿Cómo?")
    # Ejemplo de datos en DataFrame (similar al formato que mencionaste)
    st.sidebar.subheader("Ingresos de datos del ¿Cómo?:")  
    # Ingreso de datos mediante un DataFrame editable
    st.sidebar.write("Modifique los datos de acuerdo a su necesidad para el diagrama de flujo:")

    # Función para generar el diagrama de flujo con colores llamativos, texto blanco y proporciones ajustadas
    def generate_flowchart(steps):
        dot = graphviz.Digraph()

        # Colores por tipo de actividad, más llamativos
        color_map = {
            "Inicio / Fin": "#32CD32",  # LimeGreen
            "Operación / Actividad": "#FFD700",  # Gold
            "Documento": "#1E90FF",  # DodgerBlue
            "Datos": "#FF6347",  # Tomato
            "Almacenamiento / Archivo": "#8A2BE2",  # BlueViolet
            "Decisión": "#FF4500"  # OrangeRed
        }

        # Agregar nodos de inicio y fin fijos con colores llamativos y texto blanco
        dot.node("Inicio", label="Inicio", shape="oval", style="filled", fillcolor=color_map["Inicio / Fin"], fontcolor="white")
        dot.node("Fin", label="Fin", shape="oval", style="filled", fillcolor=color_map["Inicio / Fin"], fontcolor="white")

        # Agregar los pasos del diagrama de flujo
        for step in steps:
            fillcolor = color_map.get(step['tipo'], 'white')
            dot.node(step['id'], label=step['etiqueta'], shape=step['forma'], style="filled", fillcolor=fillcolor, fontcolor="white")

        # Conectar los pasos
        dot.edge("Inicio", steps[0]['id'])  # Conexión desde el inicio al primer paso
        for step in steps:
            if 'siguiente' in step:
                for next_step_id in step['siguiente']:
                    dot.edge(step['id'], next_step_id)
        dot.edge(steps[-1]['id'], "Fin")  # Conexión del último paso al fin

        # Ajustar proporciones de ancho y alto (más alto que ancho)
        dot.attr(size="6,8", ratio="fill")

        return dot

    # Ejemplo precargado utilizando todos los símbolos y colores llamativos
    def cargar_proceso_ejemplo():
        return [
            {'id': 'Paso_1', 'etiqueta': 'Iniciar proceso', 'tipo': 'Operación / Actividad', 'forma': 'rectangle', 'siguiente': ['Paso_2']},
            {'id': 'Paso_2', 'etiqueta': 'Recibir documento', 'tipo': 'Documento', 'forma': 'parallelogram', 'siguiente': ['Paso_3']},
            {'id': 'Paso_3', 'etiqueta': 'Tomar decisión', 'tipo': 'Decisión', 'forma': 'diamond', 'siguiente': ['Paso_4', 'Paso_5']},
            {'id': 'Paso_4', 'etiqueta': 'Almacenar documento', 'tipo': 'Almacenamiento / Archivo', 'forma': 'invtriangle', 'siguiente': ['Paso_6']},
            {'id': 'Paso_5', 'etiqueta': 'Procesar datos', 'tipo': 'Datos', 'forma': 'parallelogram', 'siguiente': ['Paso_6']},
            {'id': 'Paso_6', 'etiqueta': 'Finalizar proceso', 'tipo': 'Operación / Actividad', 'forma': 'rectangle', 'siguiente': []},
        ]

    # Cargar o editar un proceso de ejemplo por defecto
    usar_ejemplo = st.sidebar.checkbox('Usar proceso de ejemplo', value=True)

    if usar_ejemplo:
        pasos = cargar_proceso_ejemplo()
    else:
        pasos = []

    # Seleccionar número de pasos (si no se usa el proceso de ejemplo)
    num_pasos = st.sidebar.number_input('Número de pasos', min_value=1, max_value=20, value=len(pasos) if usar_ejemplo else 1)

    # IDs automáticos para los pasos
    ids_disponibles = [f'Paso_{i + 1}' for i in range(num_pasos)]

    for i in range(num_pasos):
        st.sidebar.subheader(f'Paso {i + 1}')

        # Datos para los pasos existentes o nuevos
        if i < len(pasos):
            paso = pasos[i]
            etiqueta = st.sidebar.text_input(f'Etiqueta para el Paso {i + 1}', value=paso['etiqueta'])
            tipo = st.sidebar.selectbox(f'Tipo de actividad del Paso {i + 1}', 
                                        ['Inicio / Fin', 'Operación / Actividad', 'Documento', 'Datos', 'Almacenamiento / Archivo', 'Decisión'], 
                                        index=['Inicio / Fin', 'Operación / Actividad', 'Documento', 'Datos', 'Almacenamiento / Archivo', 'Decisión'].index(paso['tipo']))
            siguientes_pasos = st.sidebar.multiselect(f'Pasos siguientes desde el Paso {i + 1}', ids_disponibles, default=paso['siguiente'])
        else:
            etiqueta = st.sidebar.text_input(f'Etiqueta para el Paso {i + 1}', value=f'Paso {i + 1}')
            tipo = st.sidebar.selectbox(f'Tipo de actividad del Paso {i + 1}', ['Inicio / Fin', 'Operación / Actividad', 'Documento', 'Datos', 'Almacenamiento / Archivo', 'Decisión'])
            siguientes_pasos = st.sidebar.multiselect(f'Pasos siguientes desde el Paso {i + 1}', ids_disponibles)

        # Definir la forma según el tipo de actividad
        forma_map = {
            'Inicio / Fin': 'oval',
            'Operación / Actividad': 'rectangle',
            'Documento': 'parallelogram',
            'Datos': 'parallelogram',
            'Almacenamiento / Archivo': 'invtriangle',
            'Decisión': 'diamond'
        }
        forma = forma_map.get(tipo, 'rectangle')

        # Actualizar o agregar el paso
        if i < len(pasos):
            pasos[i] = {
                'id': f'Paso_{i + 1}',
                'etiqueta': etiqueta,
                'tipo': tipo,
                'forma': forma,
                'siguiente': siguientes_pasos
            }
        else:
            pasos.append({
                'id': f'Paso_{i + 1}',
                'etiqueta': etiqueta,
                'tipo': tipo,
                'forma': forma,
                'siguiente': siguientes_pasos
            })

    # Mostrar el diagrama de flujo generado
    st.graphviz_chart(generate_flowchart(pasos))
    st.markdown("Para más información consulta en: [Flowchart](https://asq.org/quality-resources/flowchart)")

# Pestaña: Por qué
with tabs[6]:
    st.header("¿Por qué?")
    st.sidebar.subheader("Ingresos de datos del ¿Por qué?:")
    
    # Ejemplo de datos en DataFrame
    example_data = {
        "Categoría": ["Idea A", "Idea B", "Idea C", "Idea D", "Idea E", "Idea F"],
        "Variable x": [10, 15, 5, 30, 2, 60],
        "Variable y": [3, 5, 2, 10, 1, 15],
        "Variable z": [1000, 1500, 100, 500, 0, 250]
    }
    example_df = pd.DataFrame(example_data)

    # Ingreso de datos mediante un DataFrame editable
    st.sidebar.write("Modifique los datos de acuerdo a su necesidad:")
    df_porque = st.sidebar.data_editor(example_df, num_rows="dynamic", key="df_porque")

    # Selección de columnas para el gráfico
    columnas = df_porque.columns.tolist()
    
    x_col = st.sidebar.selectbox("Seleccione la variable para el eje X:", columnas, index=1)
    y_col = st.sidebar.selectbox("Seleccione la variable para el eje Y:", columnas, index=2)
    size_col = st.sidebar.selectbox("Seleccione la variable para el tamaño de las burbujas:", columnas, index=3)
    color_col = st.sidebar.selectbox("Seleccione la variable para el color de las burbujas:", columnas, index=3)

    # Verificar si los nombres ingresados son válidos en el DataFrame
    if not df_porque.empty:
        if all(col in df_porque.columns for col in [x_col, y_col, size_col, color_col]):
            # Crear gráfico de burbujas con Plotly si los nombres de las columnas son válidos
            fig = px.scatter(
                df_porque,
                x=x_col,  # Variable en el eje X
                y=y_col,  # Variable en el eje Y
                size=size_col,  # Tamaño de las burbujas
                text="Categoría",  # Etiquetas para las burbujas
                color=color_col,  # Color basado en la variable seleccionada
                title="Gráfico de Burbujas Personalizado",
                size_max=10,  # Ajuste del tamaño máximo de las burbujas
                color_continuous_scale=px.colors.sequential.Plasma  # Escala de colores
            )

            # Personalizar diseño
            fig.update_traces(marker=dict(sizemode='diameter', opacity=0.8, line=dict(width=2, color='DarkSlateGrey')),
                              selector=dict(mode='markers+text'))
            fig.update_layout(height=600, width=900)  # Ajustar tamaño del gráfico

            # Mostrar gráfico en la aplicación
            st.plotly_chart(fig)
        else:
            st.error("Algunas de las columnas ingresadas no existen en el DataFrame. Verifique los nombres.")

    st.markdown("Para más información consulta en: [Bubble chart](https://plotly.com/python/bubble-charts/)")

# Pestaña: Acerca de mí
with tabs[7]:
    #st.header("Acerca de mí")
    
    st.subheader("José Miguel Aguilar Torrealba")
    st.write("**Chemical Engineer | MBA Digital Business Administration | Innovation & CI Facilitator | Coaching | Citizen Data Scientist | Lean Six Sigma | Integrated Management Systems Auditor**")
    
    st.subheader("Resumen profesional")
    st.write("""
    20 años de experiencia en los sectores químico, educativo y de alimentos, me especializo en modelos de mejora continua, excelencia operacional y proyectos de tecnología e infraestructura. He trabajado en la industria y en el ámbito académico, facilitando y auditoriando sistemas de gestión.
    """)
    
    st.subheader("Experiencia Profesional")
    st.write("""
    - **Continuous Improvement Manager** en AkzoNobel (abr. 2022 - actualidad): Implementación de modelos globales de mejora continua, gestión de iniciativas de ahorro y transferencia de buenas prácticas.
    
    - **SR Continuous Improvement Coordinator** en Andercol SAS (feb. 2016 - abr. 2022): Diseño de estrategias de mejora, Industria 4.0 y Big Data, y gestión de la excelencia operacional regional.
    
    - **Senior Process & Project Coordinator** en C.A. Venezolana de Pinturas (feb. 2014 - ene. 2016): Control de productos y procesos, reducción del impacto ambiental y optimización de la producción.
    
    - **Process Engineer** en C.A. QUIMICA INTEGRADA -INTEQUIM- (may. 2010 - feb. 2014): Coordinación de mejoras en procesos y equipos para optimizar la productividad.
    
    - **Seminary Professor** en Universidad de Carabobo (sept. 2009 - sept. 2010): Facilitación de nuevas tendencias en ingeniería química y mejoramiento continuo.
    
    - **CI & Quality Advisor** en Palma Products International C.A. (nov. 2009 - abr. 2010): Reingeniería del SGC y mejora continua.
    
    - **Continuous Improvement Engineer** en Ajegroup (ene. 2005 - oct. 2008): Implementación de metodologías Lean Manufacturing, Kaizen, y TPM.
    
    - **Process Engineer** en Ajegroup (ago. 2006 - nov. 2006): Mejora de sistemas hidráulicos y procesos de producción.
    
    - **Quality Auditor** en Ajegroup (nov. 2005 - jul. 2006): Control y auditoría de calidad.
    
    - **Continuous Improvement Intern** en Ajegroup (ago. 2004 - nov. 2005): Participación en control estadístico de procesos y Lean Six Sigma.
    
    - **Analytical Chemistry Trainer** en Universidad de Carabobo (mar. 2004 - nov. 2005): Entrenador en química analítica y asistente del profesor.
    """)

    st.subheader("Formación")
    st.write("""
    - **Grand Master MBA Dirección de Negocio Digital**, TECH Universidad (ene. 2021 - feb. 2023)
    
    - **Ingeniero Químico**, Universidad de Carabobo (ago. 2000 - ago. 2007)
    
    - **Certificaciones y Formación Adicional**:
      - Lean Six Sigma
      - Big Data
      - Transformación Digital
      - Control de Procesos Químicos
      - Auditoría de Sistemas Integrados de Gestión
    """)
  
    # Agregar un enlace al final de la pestaña
    st.markdown("Para más información consulta en: [LinkedIn](https://www.linkedin.com/in/josemaguilar/)")
