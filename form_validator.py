from tensorflow.keras import models
import numpy as np
from signature_processor import string_to_img

cnn = models.load_model('models/model.h5')

classes = ["Tinta negra y campos llenos",
           "Tinta negra y campos vacíos",
           "Tinta de colores y campos llenos",
           "Tinta de colores y campos vacíos"]

attributes = [
    ["negra", "completos"], 
    ["negra", "incompletos"], 
    ["color", "completos"], 
    ["color", "incompletos"]]

def predict_class(string):
    img = string_to_img(string)
    npimg = np.array(img)
    
    pred = cnn.predict( np.array([npimg,]))[0]
    predicted_attributes = np.argmax(pred)

    return attributes[predicted_attributes]
