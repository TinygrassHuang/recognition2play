import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import joblib, keras.models, cv2
import mediapipe as mp
import numpy as np
from directkeys import PressKey, ReleaseKey, KEY_Q, KEY_W, KEY_E, KEY_R, KEY_D, KEY_F

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class HandGestureRecognition:
    labelEncoder = None
    minMaxScaler = None
    model = None

    def __init__(self):
        self.label_encoder = joblib.load("LabelEncoder.pkl")
        self.minMaxScaler = joblib.load("MinMaxScaler.pkl")
        self.model = keras.models.load_model("hand_gesture_classifier.h5")
        self.start()

    def start(self):
        cap = cv2.VideoCapture(0)
        windowName = "Hand Gesture Recognition"
        with mp_hands.Hands(min_detection_confidence=0.5,
                            min_tracking_confidence=0.5,
                            max_num_hands=1) as hands:
            while 1:
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = hands.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                temp_data = []
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing_styles.get_default_hand_landmarks_style(),
                            mp_drawing_styles.get_default_hand_connections_style())
                    # summarise landmarks
                    for landmark in results.multi_hand_landmarks[0].landmark:  # only 1 hand, so [0]
                        temp_data.append([landmark.x, landmark.y, landmark.z])
                    temp_data = np.array(temp_data)
                    temp_data = temp_data - temp_data[0, :]  # relative coordinate (with respect to wrist)
                    temp_data = temp_data.flatten()

                    # make prediction according to gesture landmarks
                    prediction = self.makePrediction(temp_data)
                    # press key according to prediction
                    self.pressByPrediction(prediction)

                    image_height, image_width, _ = image.shape
                    wristX = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].x * image_width
                    wristY = results.multi_hand_landmarks[0].landmark[mp_hands.HandLandmark.WRIST].y * image_height
                    cv2.putText(image, prediction,
                                org=(int(wristX), int(wristY + 30)),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=2,
                                color=(0, 0, 255),
                                thickness=4)

                # Flip the image horizontally for a selfie-view display.
                cv2.imshow(windowName, image)

                # quit method
                key = cv2.waitKey(1)
                if key == 27:  # exit on pressing ESC
                    break
                if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:  # exit on closing window
                    break
            cap.release()
            cv2.destroyAllWindows()

    # normalise raw data and make prediction,
    # add a threshold level later
    def makePrediction(self, raw_data):
        feature = self.minMaxScaler.transform(raw_data.reshape(1, -1))
        prediction = np.argmax(self.model.predict(feature, verbose=0), axis=1)
        return self.label_encoder.inverse_transform(prediction)[0]

    def pressByPrediction(self, prediction):
        if prediction == "Q":
            self.keyPressRelease(KEY_Q)
        elif prediction == "W":
            self.keyPressRelease(KEY_W)
        elif prediction == "E":
            self.keyPressRelease(KEY_E)
        elif prediction == "R":
            self.keyPressRelease(KEY_R)
        elif prediction == "D":
            self.keyPressRelease(KEY_D)
        elif prediction == "F":
            self.keyPressRelease(KEY_F)

    def keyPressRelease(self, keyCode):
        PressKey(keyCode)
        ReleaseKey(keyCode)


if __name__ == "__main__":
    app = HandGestureRecognition()
