from flask import Flask
from flask import request
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim

app = Flask(__name__)


@app.route("/")
def prueba():
    return "Prueba"


@app.route('/post', methods=['POST'])
def login():
    usuario = request.get_json()
    return "Hola, " + usuario["nombre"]

@app.route('/file', methods=['POST'])
def file():
    img1 = request.files["img1"].read()
    npimg1 = np.fromstring(img1, np.uint8)
    imagen1 = cv2.imdecode(npimg1, cv2.IMREAD_UNCHANGED)

    img2 = request.files["img2"].read()
    npimg2 = np.fromstring(img2, np.uint8)
    imagen2 = cv2.imdecode(npimg2, cv2.IMREAD_UNCHANGED)

    i1 = cv2.resize(cv2.cvtColor(imagen1.copy(),cv2.COLOR_BGR2GRAY),(100,100))
    i2 = cv2.resize(cv2.cvtColor(imagen2.copy(),cv2.COLOR_BGR2GRAY),(100,100))

    value = ssim(i1, i2, multichannel=True)

    if(value > 0.5):
        return "Las imagenes son similares"

    return "Las imagenes son diferentes"

@app.route('/compare-signatures', methods=['POST'])
def compare_signatures():
    x = ""
    pass