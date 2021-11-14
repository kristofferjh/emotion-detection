import cv2
import numpy as np
import tensorflow as tf

# model = model_from_json(open("fer.json", "r").read())
model = tf.keras.models.load_model('model_optimal.h5')

# model.load_weights('fer.h5')
model.load_weights('model_weights.h5')

face_haar_cascade = cv2.CascadeClassifier('input/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # select the default video capture

# If the camera was not opened sucessfully
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Continously read the frames
while True:
    # read frame by frame and get return whether there is a stream or not
    ret, frame = cap.read()

    # If no frames recieved, then break from the loop
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Change the frame to greyscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # We pass the image, scaleFactor and minneighbour
    faces_detected = face_haar_cascade.detectMultiScale(gray_image, 1.32, 5)

    # Draw Triangles around the faces detected
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
        roi_gray = gray_image[y: y + w, x: x + h]
        roi_gray = cv2.resize(roi_gray, (48, 48))

        # Processes the image and adjust it to pass it to the model
        image_pixels = tf.keras.preprocessing.image.img_to_array(roi_gray)
        image_pixels = np.expand_dims(image_pixels, axis=0)
        image_pixels /= 255

        # Get the prediction of the model
        predictions = model.predict(image_pixels)
        max_index = np.argmax(predictions[0])
        emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')
        emotion_prediction = emotion_detection[max_index]


        # Write on the frame the emotion detected
        cv2.putText(frame, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    resize_image = cv2.resize(frame, (1000, 700))
    cv2.imshow('Emotion', resize_image)
    if cv2.waitKey(10) == ord('b'):
        break

cap.release()
cv2.destroyAllWindows
