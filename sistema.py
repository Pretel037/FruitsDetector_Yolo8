import streamlit as st
import cv2
import numpy as np
from PIL import Image
from yolov5 import YOLOv5
import io
import tempfile
import os

import torch
base_path = os.path.join(os.getcwd(), 'Models')

# Cargar los modelos
modelpath = os.path.join(base_path, 'best.pt')

# Cargar el modelo YOLOv5
model = YOLOv5(modelpath)



st.title('Aplicación de Detección de Objetos con YOLOv5')

# Opción para capturar una imagen desde la cámara
camera_image = st.camera_input("Captura una imagen")

if camera_image is not None:
    # Muestra la imagen capturada
    image = Image.open(camera_image)
    st.image(image, caption='Imagen capturada.', use_column_width=True)
    st.write("Realizando detección...")

    # Realiza la inferencia con la imagen capturada
    results = model.predict(np.array(image))
    detections = results.pandas().xyxy[0]
    st.write("Detecciones:")
    st.dataframe(detections)

# Opción para subir una imagen
uploaded_image = st.file_uploader("Sube una imagen...", type="jpg")

if uploaded_image is not None:
    # Muestra la imagen subida
    image = Image.open(uploaded_image)
    st.image(image, caption='Imagen subida.', use_column_width=True)
    st.write("Realizando detección...")

    # Realiza la inferencia con la imagen subida
    results = model.predict(np.array(image))
    detections = results.pandas().xyxy[0]
    st.write("Detecciones:")
    st.dataframe(detections)
