
import streamlit as st
import pandas as pd

data = {
    'Modelo': ['MobileNetV2', 'Modelo X', 'Modelo Y'],
    'Precisión (%)': [97.67, 95.20, 93.50],
    'Tiempo (seg)': [1, 2, 3],
    'Tamaño (MB)': [12, 20, 18],
    'RAM (GB)': ['≤12', '16', '8'],
    'Hardware': ['Solo CPU', 'CPU + GPU', 'Solo CPU']
}

df = pd.DataFrame(data)

st.title("Tabla Comparativa de Modelos")
st.table(df)
