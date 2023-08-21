import streamlit as st
import numpy as np
import base64
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img

st.markdown('<h1 style="color:black;">Image Classification Models</h1>', unsafe_allow_html=True)

models = {
    "TRG": "bench_mark_-model-20-0.98.hdf5",
    "pT": "bench_mark_-model-19-0.92.hdf5",
    "pN": "bench_mark_-model-25-0.97.hdf5",
    "i_ven": "bench_mark_-model-25-0.91.hdf5",
    "i_perin": "bench_mark_-model-23-0.85.hdf5",
    "i_limf": "bench_mark_-model-23-0.91.hdf5",
    "grd": "bench_mark_-model-21-1.00.hdf5"
    # Add more models here
}

selected_model = st.selectbox("Select a classification model", list(models.keys()))

st.markdown(f'<h2 style="color:gray;">{selected_model} classifies images into the following categories:</h2>',
            unsafe_allow_html=True)

# load the selected model
model_filename = models[selected_model]
model = tf.keras.models.load_model(model_filename)

upload = st.file_uploader('Insert image for classification', type=['png', 'jpg'])
c1, c2 = st.columns(2)

if upload is not None:
    image = load_img(upload, target_size=(150, 150))
    image = np.asarray(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    c1.header('Input Image')
    c1.image(image, clamp=True)

    # prediction on model
    preds = model.predict(image)
    pred_classes = np.argmax(preds, axis=1)
    c2.header('Output')
    c2.subheader('Predicted class :')
    c2.write(pred_classes[0])
