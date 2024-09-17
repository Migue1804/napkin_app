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

# Configuración de la aplicación
st.set_page_config(page_title="Visualización de Marcos SCVID", layout="wide")
# Display the image above the title
st.image('Napkin App.png', use_column_width=True)

# Sidebar
st.sidebar.title("Datos de Entrada")

# Tabs para cada tipo de problema
tabs = st.tabs(["Quién/Qué", "Cuánto", "Dónde", "Cuándo", "Cómo", "Por qué"])

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
    
# Función para crear el gráfico Quién/Qué con varias capas de atributos
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
                "image": node[1].get('image', ''),
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

    # Configurar layout del grafo
    person_net.repulsion(
        node_distance=200,
        central_gravity=0.33,
        spring_length=100,
        spring_strength=0.10,
        damping=0.95
    )

    # Guardar y mostrar grafo en HTML
    path = '/tmp'
    person_net.save_graph(f'{path}/pyvis_graph.html')
    HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')

    # Mostrar grafo en la app
    components.html(HtmlFile.read(), height=600)

# Pestaña: Quién/Qué
with tabs[0]:
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
        
# Función para crear gráfico de Pareto con Plotly
def crear_grafico_pareto_plotly(datos):
    # Ordenar los datos de mayor a menor
    datos = datos.sort_values(by='Valor', ascending=False).reset_index(drop=True)

    # Calcular el valor acumulado y su porcentaje
    datos['Acumulado'] = datos['Valor'].cumsum()
    total = datos['Valor'].sum()
    datos['% Valor'] = datos['Valor'] / total * 100
    datos['% Acumulado'] = datos['Acumulado'] / total * 100

    # Crear heatmap para las barras
    colorscale = 'Inferno'
    heatmap_colors = datos['Valor'] / datos['Valor'].max()  # Normalizar valores para el colormap

    # Crear gráfico de barras
    fig = go.Figure()

    # Agregar las barras del gráfico
    fig.add_trace(go.Bar(
        x=datos['Categoría'],
        y=datos['% Valor'],
        marker=dict(
            color=heatmap_colors,
            colorscale=colorscale,
            showscale=True,  # Mostrar barra de colores
            colorbar=dict(title="Valor")
        ),
        name='Porcentaje del Valor',
    ))

    # Agregar la línea de porcentaje acumulado
    fig.add_trace(go.Scatter(
        x=datos['Categoría'],
        y=datos['% Acumulado'],
        mode='lines+markers',
        marker=dict(color='red', symbol='diamond'),
        line=dict(color='red', width=2),
        name='Porcentaje Acumulado',
    ))

    # Agregar línea del 80%
    fig.add_shape(
        type="line",
        x0=0, x1=1, y0=80, y1=80,
        line=dict(color="gray", width=2, dash="dash"),
        xref="paper", yref="y2"  # Línea horizontal en el segundo eje
    )

    # Configurar el diseño del gráfico
    fig.update_layout(
        title='Gráfico de Pareto',
        xaxis=dict(title='Categoría', tickangle=-45),
        yaxis=dict(title='Porcentaje del Valor', range=[0, 100], showgrid=False),
        yaxis2=dict(title='Porcentaje Acumulado', overlaying='y', side='right', range=[0, 100]),
        height=600,
        legend=dict(x=0.85, y=1.15),
    )

    # Mostrar el gráfico en la app Streamlit
    st.plotly_chart(fig)

# Pestaña: '¿Cuánto?'
with tabs[3]:
    st.header("¿Cuánto?")
    st.sidebar.subheader("Datos del gráfico de Pareto:")

    # Entrada de datos para el gráfico de Pareto en el sidebar
    st.sidebar.write("Ingrese los datos para generar el gráfico de Pareto:")

    # Crear un DataFrame editable para ingresar los datos
    example_data = {
        'Categoría': ['A', 'B', 'C', 'D', 'E', 'F'],
        'Valor': [50, 30, 20, 10, 5, 3]
    }
    
    # Crear DataFrame con datos de ejemplo
    df_pareto = pd.DataFrame(example_data)
    
    # Data editor en el sidebar
    df_pareto = st.sidebar.data_editor(df_pareto, num_rows="dynamic", key="df_pareto")

    # Mostrar gráfico solo si hay datos
    if not df_pareto.empty:
        crear_grafico_pareto_plotly(df_pareto)

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
with tabs[2]:
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
    fig = px.timeline(
        eventos,
        x_start="Fecha de Inicio", 
        x_end="Fecha de Fin", 
        y="Evento", 
        color='Categoría',  # Esto es opcional si deseas agregar categorías para colorear las tareas
        title='Gráfico de Gantt de Eventos'
    )

    fig.update_yaxes(autorange="reversed")  # Invertir el eje y para que las tareas se vean de arriba hacia abajo
    fig.update_layout(height=600, width=900)  # Ajustar el tamaño del gráfico
    st.plotly_chart(fig)

with tabs[3]:
    st.header("¿Cuándo?")

    # Ejemplo de datos en DataFrame (similar al formato que mencionaste)
    st.sidebar.subheader("Ingresos de datos del ¿Cuándo?:")
    example_data = {
        "Evento": ["Proyecto Inicio", "Desarrollo", "Fin"],
        "Fecha de Inicio": ["2023-01-01", "2023-06-01", "2023-09-01"],
        "Fecha de Fin": ["2023-06-01", "2023-09-01", "2023-12-01"],
        "Categoría": ["Planificación", "Ejecución", "Cierre"]
    }
    example_df = pd.DataFrame(example_data)
    #st.sidebar.write(example_df)

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

# Pestaña: Cómo (modificada para diagrama de flujo con nodos de inicio y fin)
with tabs[4]:
    st.header("¿Cómo?")
    # Ingreso de datos de las actividades en el sidebar
    st.sidebar.subheader("Ingresos de datos del ¿Cómo?:")
    
    # Ejemplo de datos de actividades y tipos
    example_data = {
        "Actividad": ["Paso 1", "Paso 2", "Paso 3","Paso 4","Paso 5","Paso 6","Paso 7","Paso 8"],
        "Tipo": ["Almacenamiento","Set-up","Operación","Inspección", "Espera","Transporte","Almacenamiento","Transporte"],
        "Simultáneo con": ["", "Paso 1", "","Paso 3","","","",""]
    }
    df_actividades = pd.DataFrame(example_data)

    # Ingreso de actividades, tipo y si ocurre simultáneamente
    st.sidebar.write("Ingrese las actividades, tipo y si ocurre simultáneamente:")
    df_actividades = st.sidebar.data_editor(df_actividades, num_rows="dynamic", key="df_como")

    # Mapeo de tipos de actividad a URLs de imágenes (PNG)
    tipo_imagen = {
        "Set-up": "https://img.icons8.com/fluency/48/000000/maintenance.png",
        "Operación": "https://img.icons8.com/fluency/48/000000/settings.png",
        "Transporte": "https://img.icons8.com/fluency/48/000000/truck.png",
        "Inspección": "https://img.icons8.com/fluency/48/000000/inspection.png",
        "Almacenamiento": "https://img.icons8.com/fluency/48/000000/warehouse.png",
        "Espera": "https://img.icons8.com/fluency/48/000000/hourglass.png"
    }

    # Crear gráfico de Network si hay actividades
    if not df_actividades.empty:
        # Crear un gráfico de Network dirigido
        G = nx.DiGraph()

        # Agregar nodo de inicio (verde) con imagen
        G.add_node(
            "Inicio",
            label="Inicio",
            color="green",
            image="https://img.icons8.com/fluency/48/000000/play-circle.png"
        )

        # Agregar nodos para las actividades con imágenes
        for idx, row in df_actividades.iterrows():
            actividad = row["Actividad"]
            tipo = row["Tipo"]
            imagen = tipo_imagen.get(
                tipo,
                "https://img.icons8.com/fluency/48/000000/circle.png"  # Imagen por defecto
            )
            G.add_node(
                actividad,
                label=actividad,
                image=imagen,
                color="lightblue"  # Color predeterminado para actividades
            )

        # Agregar nodo de fin (rojo) con imagen
        G.add_node(
            "Fin",
            label="Fin",
            color="red",
            image="https://img.icons8.com/fluency/48/000000/stop-circle.png"
        )

        # Agregar edges entre actividades consecutivas o simultáneas
        for idx, row in df_actividades.iterrows():
            actividad_actual = row["Actividad"]
            if idx == 0:
                # Conectar nodo de inicio al primer paso
                G.add_edge("Inicio", actividad_actual)

            if idx < len(df_actividades) - 1:
                actividad_siguiente = df_actividades.iloc[idx + 1]["Actividad"]
                G.add_edge(actividad_actual, actividad_siguiente)
            else:
                # Conectar el último nodo al nodo de fin
                G.add_edge(actividad_actual, "Fin")

            # Agregar conexión si es simultáneo con otra actividad
            if row["Simultáneo con"]:
                G.add_edge(actividad_actual, row["Simultáneo con"])

        # Crear visualización con PyVis
        flow_net = Network(
            height='600px',
            width='100%',
            bgcolor='#222222',
            font_color='white',
            directed=True
        )

        # Agregar nodos y edges al gráfico
        for node in G.nodes(data=True):
            node_options = {
                "label": node[1]['label'],
                "shape": "circularImage",
                "image": node[1].get('image', ''),
                "color": node[1].get('color', 'lightblue')
            }
            flow_net.add_node(node[0], **node_options)

        for edge in G.edges():
            flow_net.add_edge(edge[0], edge[1])

        # Configurar el layout del gráfico
        flow_net.repulsion(
            node_distance=200,
            central_gravity=0.33,
            spring_length=100,
            spring_strength=0.10,
            damping=0.95
        )

        # Guardar y mostrar el gráfico en HTML
        path = '/tmp'
        flow_net.save_graph(f'{path}/pyvis_flow_diagram.html')
        HtmlFile = open(f'{path}/pyvis_flow_diagram.html', 'r', encoding='utf-8')
        components.html(HtmlFile.read(), height=600)

    # Leyenda ajustada para que coincida con las imágenes
    st.markdown("### Tipos de Actividad:")
    st.markdown('- **Set-up:** <img src="https://img.icons8.com/fluency/48/000000/maintenance.png" width="30"/>', unsafe_allow_html=True)
    st.markdown('- **Operación:** <img src="https://img.icons8.com/fluency/48/000000/settings.png" width="30"/>', unsafe_allow_html=True)
    st.markdown('- **Transporte:** <img src="https://img.icons8.com/fluency/48/000000/delivery.png" width="30"/>', unsafe_allow_html=True)
    st.markdown('- **Inspección:** <img src="https://img.icons8.com/fluency/48/000000/inspection.png" width="30"/>', unsafe_allow_html=True)
    st.markdown('- **Almacenamiento:** <img src="https://img.icons8.com/fluency/48/000000/warehouse.png" width="30"/>', unsafe_allow_html=True)
    st.markdown('- **Espera:** <img src="https://img.icons8.com/fluency/48/000000/hourglass.png" width="30"/>', unsafe_allow_html=True)


# Pestaña: Por qué
with tabs[5]:
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
