import numpy as np
import cv2
import tensorflow as tf

# Load the model
myModel = tf.keras.models.load_model('modelZoom_Fliph.h5')


# Preprocessing frame 
def modelResult(frame):
    #OpenCv uses BGR and the model is trained on RGB

    # Resizing, adding a dimension and changing BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame,(128,128),3)
    #frame = cv2.resize(frame,(150,150),3)
    frame = np.expand_dims(frame, axis=0)

    #Here we get the result 
    result = myModel.predict(frame).argmax(axis=1).astype(int)
    classes = ['forest', 'fire', 'smoke']
    return classes[result[0]]
    

#Open the video capture with a 0 to use the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    mensaje = modelResult(frame)
    #Creating the frame with the result as a text
    image = cv2.putText(frame,mensaje, (200,200), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,0),10)
    #Displaying the new frame with the text
    cv2.imshow("Modelo de deteccion de incendios",image)
    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()