# -*- coding: utf-8 -*-
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.models import load_model
	
modelNP = load_model("C:\\Users\\88691\\Downloads\\ChestX-Ray2_2313121_acc95_rec97_prec95.h5")
modelBV = load_model("C:\\Users\\88691\\Downloads\\ChestX-Ray2BV_23141627_acc92_rec89_prec89.h5")

def upload_and_recognize():
    filepath = tk.filedialog.askopenfilename(filetypes=[('jpeg', '*.jpeg'),('png', '*.png'),('jpg', '*.jpg'),('gif', '*.gif')]) 
    if not filepath:
        return

    # 讀取圖片文件並使用 PIL 將其轉換為 PhotoImage 對象
    img = Image.open(filepath)
    img = img.resize((500,500))
    w, h = img.size
    image = ImageTk.PhotoImage(img)
    canvas.delete('all')                
    canvas.config(scrollregion=(0,0,w,h))
    canvas.create_image(0, 0, anchor='nw', image=image)  
    canvas.image = image   
          
    try:
        # 讀取圖像並預處理
        data = np.ndarray(shape=(1, 255, 255, 3), dtype = np.float32)
        img = img.resize((255,255))
		#turn the image into a numpy array
        image_array = np.stack((img,)*3, axis=-1)
        #pretrained前處理
        preprocessing_funct = tf.keras.applications.efficientnet_v2.preprocess_input
        image_array = preprocessing_funct(image_array)
        data[0] = image_array.astype(np.float32)/255.0
        resultNP = modelNP.predict(data)
		# Deleting previos result
        text.delete(1.0, tk.END)
        
        prob_str = []
        for prob in resultNP[0]:
            if prob > 0.3:
                data = np.ndarray(shape=(1, 256, 256, 3), dtype=np.float32)
                img = img.resize((256,256))
        		#turn the image into a numpy array
                image_array = np.stack((img,)*3, axis=-1)
                #pretrained前處理
                normalized_image_array = image_array.astype(np.float32)/255.0
                data[0] = normalized_image_array
                # 使用模型進行圖像辨識
                resultBV = modelBV.predict(data)
                prob_str = []
                for prob in resultBV[0]:
                    prob_str.append("Viral Pneumonia : {:.2f}%".format(prob * 100) if prob > 0.5 else "bacterial Pneumonia : {:.2f}%".format(prob * 100))
            else:
                prob_str.append("Healthy Lungs : {:.2f}%".format(prob * 100))

        text.insert('end', filepath.split('/')[-1]+"\n")
        text.insert('end', ', '.join(prob_str))
        
    except Exception as e:
        text.insert('end', str(e))

root = tk.Tk()
root.title('PNEUMONIA Test')

label1 = tk.Label(root, 
                  pady=20,
                  text='X-RAY NORMAL/PNEUMONIA',
                  font=('DIN Alternate Light',20),
                  fg='#BAF8FF',
                  bg='#2D4D67')
label1.pack(fill='x',side='top')

frame = tk.Frame(root, bd=10, relief="sunken", bg='#2D4D67')                  
frame.pack()

canvas = tk.Canvas(frame, width=500, height=450, bg='#fff')
canvas.pack(side='left')

button = tk.Button(root, text = 'Upload & Recognize', command = upload_and_recognize)
button.pack()

label = tk.Label(root)
label.pack()

text = tk.Text(root)
text.pack()

root.mainloop()
