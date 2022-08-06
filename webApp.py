import streamlit as st
import numpy as np
from PIL import Image 
from tensorflow import keras
from keras import models
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing import image
import cv2
nltk.download('stopwords')
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

def load_image(image_file):
    img = Image.open(image_file)
    return img

def pred_result(img):
    lesion_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
        }
    lesion_class_dict = {
        0 : 'nv',
        1 : 'mel',
        2 : 'bkl',
        3 : 'bcc',
        4 : 'akiec',
        5 : 'vasc',
        6 : 'df',
        }
    cancer_class = {
        0 : 'Non-Cancerous',
        1 : 'Cancerous',
        2 : 'Non-Cancerous',
        3 : 'Cancerous',
        4 : 'Cancerous',
        5 : 'Non-Cancerous',
        6 : 'Non-Cancerous',
        }

    new_model = tf.keras.models.load_model("C:\\Users\\risha\\OneDrive\\Desktop\\Model_minor_2")

    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis = 0)
    img_preprocessed = preprocess_input(img_batch)
    preds = new_model.predict(img_preprocessed)
    pred_class = np.argmax(preds, axis = -1)
    ab=0
    prt = lesion_class_dict[pred_class[ab]]
    result = str(prt)
    return [lesion_dict[result], cancer_class[pred_class[ab]]]
    

def capture():
    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)
    cv2.namedWindow("test")
    img_counter = 0


    #run = st.checkbox('Run')
    while True:
        _, frame = camera.read()
        cv2.imshow("test", frame)
        k = cv2.waitKey(1)
        if k%256 == 32:
            #space pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1
            
        elif k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
    #st.write(type(frame))
    image_view = cv2.imread(r"C:\\Users\\risha\\opencv_frame_0.png")
    image_view_arr = np.asarray(image_view)
    #st.write(type(image_view_arr))
    frame = cv2.cvtColor(image_view_arr, cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(frame)



def main():
    
    st.title("Skin Cancer Prediction")
    menu = ["How to use", "Image Capture", "Image Upload", "Exploratory Data Analysis", "About"]
    choice = st.sidebar.selectbox("Menu",menu)
 #   if choice == "EDA":
  #       st.subheader("EDA")
   #  elif choice == "Image Upload":
    #     st.subheader("Image Upload")
     # elif choice == "Image Capture":
      #    st.subheader("Image Capture")


    if choice == "How to use":
        st.subheader("How to use")
        st.write("To capture the image : Press space")
        st.write("To exit the webcam window : Press esc")
        

    elif choice == "Exploratory Data Analysis":
        st.subheader("Exploratory Data Analysis")
        df = pd.read_csv("C:\\Users\\risha\\OneDrive\\Desktop\\HAM.csv", nrows = 1000)
        #st.write(df)
        st.bar_chart(df['localization'])
        df = pd.read_csv("C:\\Users\\risha\\OneDrive\\Desktop\\HAM.csv")
        x = df['age']
        y = df['cell_type_idx']
        fig=plt.figure(figsize = (4,3))
        plt.scatter(x,y)
        plt.xlabel("AGE")
        plt.ylabel("Cell_type")
        #st.baloons()
        st.pyplot(fig)

        
    elif choice == "Image Upload":
        st.subheader("Image Upload")
        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
        if image_file is not None:
            # To See details
            file_details = {"filename":image_file.name, "filetype":image_file.type,
                            "filesize":image_file.size}
            st.write(file_details)
            # To View Uploaded Image
            img = load_image(image_file)
            st.image(img,width=250)
            img.save(r'C:\Users\risha\img_updated.png')
            img = image.load_img(r'C:\Users\risha\img_updated.png', target_size = (28,28,3))
            
            res = pred_result(img)
            st.write('You are suffering from : ', res[0])
            st.write('Type : ', res[1])
            
    elif choice == "Image Capture":
        st.subheader("Webcam Live Feed")
        
        capture()
        
        image_path = r"C:\\Users\\risha\\opencv_frame_1.png"
        img = image.load_img(image_path, target_size = (28,28,3))

        res = pred_result(img)
        st.write('You are suffering from : ', res[0])
        st.write('Type : ', res[1])
                    
    elif choice == "About":
        st.subheader("About")
        st.text("Skin diseases are a very common issue for any human and occur due to the fact \nthat the skin is exposed freely to the outside world. Now the onset of \nartificial intelligence and machine learning techniques, in the field of \nimages, has allowed computers to identify sequences and patterns in images that \ncan never be observed by the naked eye. Hence in order to battle skin cancer in \nits early stages a system has been proposed to identify and predict skin cancer \nin its earlier stages. A skin cancer prediction system has hence been \ncreated and implemented to predict three major types of skin cancer that affect \nhumans.")        
        st.text("1. Melanoma \n2. Basal Cell Carcinoma \n3. Actinic Keratosis")
        st.text("This project’s main aim is to provide a fast and accurate diagnosis of \nskin cancer for anyone using the web application. It is essentially free of cost \nand can provide a great help to people in remote areas using this, who have \nlimited access to proper healthcare.")

main()










