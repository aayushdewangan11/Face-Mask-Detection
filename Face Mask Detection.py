import cv2
import h5py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

model = load_model(r"C:/Users/aayus/Desktop/CV/model_acc_974_vacc_991.h5")

img_width, img_hight = 200, 200

face_cascade = cv2.CascadeClassifier(r"C:\Users\aayus\Desktop\CV\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(r"C:\Users\aayus\Downloads\Video\Mask - 34775.mp4") # for video

img_count_full = 0

#parameters for text
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 
# org 
org = (1, 1)
class_lable=' '      
# fontScale 
fontScale = 1 #0.5
# Blue color in BGR 
color = (255, 0, 0) 
# Line thickness of 2 px 
thickness = 2 #1

#sart reading images and prediction
while True:
    img_count_full += 1
    
    #read image from webcam
    responce, color_img = cap.read()
    #color_img = cv2.imread('sandeep.jpg')
    
    #if respoce False the break the loop
    if responce == False:
        break    
    


    
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray_img, 1.1, 6) # 1.1, 3) for 1.mp4
    
    #take face then predict class mask or not mask then draw recrangle and text then display image
    img_count = 0
    for (x, y, w, h) in faces:
        org = (x-10,y-10)
        img_count +=1 
        color_face = color_img[y:y+h,x:x+w] # color face
        cv2.imwrite(r'C:\Users\aayus\Desktop\CV\Input Faces/%d%dface.jpg'%(img_count_full,img_count),color_face)
        img = load_img(r'C:\Users\aayus\Desktop\CV\Input Faces/%d%dface.jpg'%(img_count_full,img_count), target_size=(img_width,img_hight))
        
        img = img_to_array(img)/255
        img = np.expand_dims(img,axis=0)
        pred_prob = model.predict(img)
        #print(pred_prob[0][0].round(2))
        pred=np.argmax(pred_prob)
            
        if pred==0:
            print("User with mask - predic = ",pred_prob[0][0])
            class_lable = "Mask"
            color = (255, 0, 0)
            cv2.imwrite(r'C:\Users\aayus\Desktop\CV\Input Faces/%d%dface.jpg'%(img_count_full,img_count),color_face)
                 
        else:
            print('user not wearing mask - prob = ',pred_prob[0][1])
            class_lable = "No Mask"
            color = (0, 255, 0)
            cv2.imwrite(r'C:\Users\aayus\Desktop\CV\Input Faces/%d%dface.jpg'%(img_count_full,img_count),color_face)
                
        cv2.rectangle(color_img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        # Using cv2.putText() method 
        cv2.putText(color_img, class_lable, org, font,  
                                   fontScale, color, thickness, cv2.LINE_AA) 
    
    # display image
    cv2.imshow('LIVE face mask detection', color_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()