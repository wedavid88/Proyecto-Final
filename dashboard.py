import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Configuración del diseño del dashboard
st.set_page_config(page_title="Proyecto 3", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f0f8ff;}
    h1 {color: #013a63; text-align: center;}
    </style>
    """, unsafe_allow_html=True)

# =====================================
# 1. Limpieza de datos
# =====================================
st.title("Dashboard Smoking & Drinking")
st.header("Limpieza de datos")

# Cargar los datos
datos = pd.read_csv("smoking_driking_dataset.csv")

# Seleccionar únicamente las columnas numéricas
datos_numericos = datos.select_dtypes(include=['float64', 'int64'])

# Filtrado de percentiles
percentiles = datos_numericos.quantile([0.01, 0.99])
datos_limpios = datos_numericos[(datos_numericos >= percentiles.loc[0.01]) & 
                                (datos_numericos <= percentiles.loc[0.99])].dropna()
datos_limpios["sex"] = datos["sex"]
datos_limpios["DRK_YN"] = datos["DRK_YN"]
datos_limpios["SMK_stat_type_cd"] = datos["SMK_stat_type_cd"]

st.write("Datos filtrados según el percentil 1% y 99%:")
st.dataframe(datos_limpios)

# =====================================
# 2. Datos generales
# =====================================
st.header("Datos Generales")
c1, c2, c3 = st.columns(3)

with c1:
    total_hombres = datos_limpios[datos_limpios['sex'] == 'Male'].shape[0]
    st.metric("Hombres", total_hombres)

with c2:
    total_mujeres = datos_limpios[datos_limpios['sex'] == 'Female'].shape[0]
    st.metric("Mujeres", total_mujeres)

with c3:
    total_poblacion = datos_limpios.shape[0]
    st.metric("Población Total", total_poblacion)

st.subheader("Estadística Descriptiva")
st.dataframe(datos_limpios.describe())

# =====================================
# 3. Visualización: Histogramas
# =====================================
st.header("Histogramas")
variable_hist = st.selectbox("Selecciona la variable para el histograma:", datos_limpios.columns)

c1, c2, c3 = st.columns(3)
with c1:
    fig = px.histogram(datos_limpios, x=variable_hist, color="sex", title="Por Sexo", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.histogram(datos_limpios, x=variable_hist, color="DRK_YN", title="Por Categoría de Bebedor", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

with c3:
    fig = px.histogram(datos_limpios, x=variable_hist, color="SMK_stat_type_cd", title="Por Estado de Fumador", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)

# =====================================
# 4. Visualización: Gráficos de violín
# =====================================
st.header("Gráficos de Violín")
variable_violin = st.selectbox("Selecciona la variable para los gráficos de violín:", datos_limpios.columns)

c1, c2, c3 = st.columns(3)
with c1:
    fig = px.violin(datos_limpios, x="sex", y=variable_violin, color="sex", box=True, title="Por Sexo")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.violin(datos_limpios, x="DRK_YN", y=variable_violin, color="DRK_YN", box=True, title="Por Categoría de Bebedor")
    st.plotly_chart(fig, use_container_width=True)

with c3:
    fig = px.violin(datos_limpios, x="SMK_stat_type_cd", y=variable_violin, color="SMK_stat_type_cd", box=True, title="Por Estado de Fumador")
    st.plotly_chart(fig, use_container_width=True)

# =====================================
# 5. Modelo de Árbol de Decisión
# =====================================
st.header("Modelo de Árbol de Decisión: Predicción de Bebedores")

# Agregar Tabla de Clasificación por Género y Consumo de Alcohol
st.subheader("Clasificación de Personas por Género y Consumo de Alcohol")

# Crear la tabla de clasificación
clasificacion = datos_limpios.groupby(['sex', 'DRK_YN']).size().unstack(fill_value=0)

# Renombrar filas y columnas para claridad
clasificacion.index = ["Femenino (0)", "Masculino (1)"]
clasificacion.columns = ["No Bebedor (0)", "Bebedor (1)"]

# Mostrar la tabla en Streamlit
st.write("**Cantidad de hombres y mujeres que beben y no beben:**")
st.table(clasificacion)

# Traducción de las variables
traducciones = {
    "sex": "Género (0=Femenino, 1=Masculino)",
    "age": "Edad (años, múltiplos de 5)",
    "height": "Altura (cm)",
    "weight": "Peso (kg)",
    "waistline": "Circunferencia de cintura (cm)",
    "sight_left": "Visión del ojo izquierdo",
    "sight_right": "Visión del ojo derecho",
    "hear_left": "Audición oído izquierdo (1=normal, 2=anormal)",
    "hear_right": "Audición oído derecho (1=normal, 2=anormal)",
    "SBP": "Presión sistólica (mmHg)",
    "DBP": "Presión diastólica (mmHg)",
    "BLDS": "Glucosa en sangre en ayunas",
    "tot_chole": "Colesterol total (mg/dL)",
    "HDL_chole": "Colesterol HDL (mg/dL)",
    "LDL_chole": "Colesterol LDL (mg/dL)",
    "triglyceride": "Triglicéridos (mg/dL)",
    "hemoglobin": "Hemoglobina (g/dL)",
    "urine_protein": "Proteína en orina",
    "serum_creatinine": "Creatinina en sangre (mg/dL)",
    "SGOT_AST": "Transaminasa AST (IU/L)",
    "SGOT_ALT": "Transaminasa ALT (IU/L)",
    "gamma_GTP": "Gamma-GTP (IU/L)",
    "SMK_stat_type_cd": "Estado de fumador (1=Nunca, 2=Exfumador, 3=Fumador)",
    "DRK_YN": "Bebedor (Sí=1, No=0)"
}

# Preprocesamiento de datos
label_encoders = {}
for column in ['sex', 'SMK_stat_type_cd', 'DRK_YN']:
    le = LabelEncoder()
    datos_limpios[column] = le.fit_transform(datos_limpios[column])
    label_encoders[column] = le

# Variables independientes y dependientes
X = datos_limpios.drop(columns=['DRK_YN'])
y = datos_limpios['DRK_YN']

# División de los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenamiento del modelo
modelo = DecisionTreeClassifier(random_state=42, max_depth=8, max_leaf_nodes=8)
modelo.fit(X_train, y_train)

# Evaluación del modelo
y_pred = modelo.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Exactitud del modelo: {accuracy:.2f}")

# Interfaz para predicción
st.subheader("Ingresa los valores para predecir si eres bebedor:")

# Crear una cuadrícula
input_data = {}
columns = list(X.columns)
grid_data = []

for i in range(0, len(columns), 5):
    cols = st.columns(5)
    for j, col in enumerate(columns[i:i+5]):
        if col == "SMK_stat_type_cd":
            input_data[col] = cols[j].selectbox(f"{traducciones[col]}", [1, 2, 3])
        elif col == "sex":
            input_data[col] = cols[j].selectbox(f"{traducciones[col]}", [0, 1])
        else:
            input_data[col] = cols[j].number_input(f"{traducciones[col]}",
                                                   min_value=float(X[col].min()),
                                                   max_value=float(X[col].max()))
        grid_data.append(input_data[col])

# Botón para realizar la predicción
if st.button("Predecir"):
    input_df = pd.DataFrame([input_data])
    prediction = modelo.predict(input_df)
    resultado = label_encoders['DRK_YN'].inverse_transform(prediction)
    st.write(f"Resultado: {'Bebedor' if resultado[0] else 'No Bebedor'}")