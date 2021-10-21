import numpy as np
import cv2
import imutils
from skimage.metrics import structural_similarity as ssim

def string_to_img(string):
    npimg = np.fromstring(string, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    return img

def edge(img):
    npimg = np.array(img)
    ratio = npimg.shape[0]/500.0
    orig = npimg.copy() #converting to grayscale image
    gray = cv2.cvtColor(npimg,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    edged = cv2.Canny(gray,80,200)
    return edged

def get_coords(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key= cv2.contourArea, reverse=True)[:1]
    for c in cnts:
        peri = cv2.arcLength(c,True)
        apprx = cv2.approxPolyDP(c, 0.2*peri, True)
        x,y,w,h = cv2.boundingRect(apprx)
        return x, y, w, h

def crop(edged, line_width, x,y,w,h):
    cropped_image = edged[y+line_width:y+h-line_width,x+line_width:x+w-line_width]
    return cropped_image


def remove_white_space(cropped_image):
    #gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cropped_image, (25,25), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    noise_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    close = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, close_kernel, iterations=3)# Find enclosing boundingbox and crop ROI\n”,
    coords = cv2.findNonZero(close)
    x,y,w,h = cv2.boundingRect(coords)
    return cropped_image[y:y+h, x:x+w]

def compare(img1, img2):
    #rsi1 = cv2.resize(cv2.cvtColor(img1.copy(),cv2.COLOR_BGR2GRAY),(100,100))
    #rsi2 = cv2.resize(cv2.cvtColor(img2.copy(),cv2.COLOR_BGR2GRAY),(100,100))
    rsi1 = cv2.resize(img1,(100,100))
    rsi2 = cv2.resize(img2,(100,100))
    
    value = ssim(rsi1, rsi2)
    return value

def compare_signatures(str1, str2):
    
    img1 = string_to_img(str1)
    img2 = string_to_img(str2)

    edged1 = edge(img1)
    edged2 = edge(img2)

    line_width = 10
    x1, y1, w1, h1 = get_coords(edged1)
    x2, y2, w2, h2 = get_coords(edged2)

    crop1 = crop(edged1, line_width, x1, y1, w1, h1)
    crop2 = crop(edged2, line_width, x2, y2, w2, h2)

    pimg1 = remove_white_space(crop1)
    pimg2 = remove_white_space(crop2)

    value = compare(pimg1, pimg2)

    return value >= 0.5
