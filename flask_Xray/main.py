# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
from PIL import Image, ImageOps #Install pillow instead of PIL
from glob import glob
from statistics import mean
from keras.models import load_model
import os 
import numpy as np
import tensorflow as tf
from flask_ngrok import run_with_ngrok

app = Flask(__name__, template_folder='templates', static_folder='upload')
#Define allowed files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
run_with_ngrok(app)

@app.route('/')  
def upload():  
    return render_template("upload.html")  

@app.route('/success', methods = ['POST'])  
def img_prediction(img_path, size, model, preprocessing_funct):
    image = Image.open(img_path).convert('RGB')
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    data = np.ndarray([1]+size+[3], dtype = np.float32)
    image_array = np.asarray(image) #turn the image into a numpy array
    image_array = preprocessing_funct(image_array) if preprocessing_funct  else image_array
    data[0] = image_array.astype(np.float32) / 255.0
    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]
    return index, confidence_score

def success():  
    NPthreshold = .3
    BVthreshold = .5
    if request.method == 'POST':  
        f = request.files['file']  
        f.save('upload/'+f.filename)  
        img_file = os.path.join('upload', f.filename)
        BVmodels = [path for path in glob(r"model\*.h5") if "BV" in path]
        NPmodels = [path for path in glob(r"model\*.h5") if "BV" not in path]
        confidence_scores=[]
        
        for NP in NPmodels:
            modelNP = load_model(NP)
            if "eff" in NP:
                size, preprocessing_funct = (255, 255),tf. keras.applications.efficientnet_v2.preprocess_input
            elif "cnn" in NP:
                size, preprocessing_funct = (128, 128), None
            elif "incv" in NP:
                size, preprocessing_funct = (128, 128), tf.keras.applications.inception_v3.preprocess_input
            elif "resnet" in NP:
                size, preprocessing_funct = (224, 224), tf.keras.applications.resnet_v2.preprocess_input
            elif "mobilenet" in NP:
                size,preprocessing_funct = (224, 224), tf.keras.applications.mobilenet.preprocess_input
            _, cf_sc = img_prediction(img_file, size, modelNP, preprocessing_funct)
            confidence_scores.append(cf_sc)
            
        probx = mean(confidence_scores)	
        if probx < NPthreshold:
            label = "NORMAL"
        else:
            confidence_scoresBV=[]
            for BV in BVmodels:
                modelBV = load_model(BV)
                if "cnn" in BV:
                    size,preprocessing_funct = (256, 256), None
                elif "resnet" in BV:
                    size,preprocessing_funct = (224, 224), tf.keras.applications.resnet_v2.preprocess_input
                elif "mobilenet" in BV:
                    size,preprocessing_funct = (224, 224), tf.keras.applications.mobilenet.preprocess_input
                elif "incv" in BV:
                    size, preprocessing_funct = (128, 128), tf.keras.applications.inception_v3.preprocess_input
                elif "eff" in BV:
                    size, preprocessing_funct = (255, 255), tf. keras.applications.efficientnet_v2.preprocess_input
                    
                _, cf_scBV = img_prediction(img_file, size, modelBV, preprocessing_funct)
                confidence_scoresBV.append(cf_scBV)
                
            proby = mean(confidence_scoresBV)		
            label = "VIRUS" if proby > BVthreshold else "BACTERIA"
            
        print(f'Class : {label}\nPneumonia Probibility : {round(100*probx,2)}')
        return render_template("success.html", name = img_file, class_name = label, probability= round(100*probx, 2))          

app.run(port=3100 ,debug = True) 