from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from signature_processor import *
from form_validator import *

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
def cosig():
    str1 = request.files["formulario"].read()
    str2 = request.files["firma"].read()
    return "Hola"



@app.route('/validate-form', methods=['POST'])
def validate_form():
    str_formulario = request.files["formulario"].read()
    str_firma = request.files["firma"].read()

    similar = compare_signatures(str_formulario, str_firma)
    firma = "diferente"
    if(similar):
        firma = "similar"

    [tinta, campos] = predict_class(str_formulario)

    print(similar)
    print(tinta)
    print(campos)

    response = {
        'firma': firma, 
        'tinta': tinta,
        'campos': campos}

    return jsonify(response)