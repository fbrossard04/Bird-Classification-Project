import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load your trained model
model = load_model('C:/Users/User/Documents/Data science/Module 7 Machine learning/Project 4/fine_tuned_model.h5')

# Create the ImageDataGenerator
unseen_datagen = ImageDataGenerator(rescale=1./255)

# Class indices
class_indices = {
    0: 'AMERICAN GOLDFINCH',
    1: 'AMERICAN ROBIN',
    2: 'BLACK-CAPPED CHICKADEE',
    3: 'CEDAR WAXWING',
    4: 'CHIPPING SPARROW',
    5: 'COMMON GRACKLE',
    6: 'DARK EYED JUNCO',
    7: 'DOWNY WOODPECKER',
    8: 'HOUSE SPARROW',
    9: 'MALLARD DUCK',
    10: 'MOURNING DOVE',
    11: 'NORTHERN CARDINAL',
    12: 'NORTHERN FLICKER',
    13: 'PURPLE FINCH',
    14: 'TREE SWALLOW'
}

def predict_image_class(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(150, 150))  # Replace with your image size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Apply the same preprocessing
    img_array = unseen_datagen.standardize(img_array)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)

    # Map the predicted class to the class label
    predicted_label = class_indices[predicted_class[0]]

    return predicted_label

# Streamlit app
st.title('Bird Classification project')

st.header('This model was trained only to identify those birds seen in Quebec:')
st.write("BLACK-CAPPED CHICKADEE", "MALLARD DUCK", "AMERICAN ROBIN", "AMERICAN GOLDFINCH", 
    "NORTHERN CARDINAL", "DOWNY WOODPECKER", "MOURNING DOVE", "HOUSE SPARROW", 
    "COMMON GRACKLE", "DARK EYED JUNCO", "CHIPPING SPARROW", "NORTHERN FLICKER", 
    "CEDAR WAXWING", "TREE SWALLOW", "PURPLE FINCH")


uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict the class of the uploaded image
    label = predict_image_class("temp.jpg")

    # Display the image and the prediction
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write(f'Predicted Label: {label}')

# Credit Section
st.header('Credits')
st.write('This project was developed using the dataset:')
st.write('https://www.kaggle.com/datasets/gpiosenka/100-bird-species')
st.write('Project developped by Victor, Luca and Francois')
