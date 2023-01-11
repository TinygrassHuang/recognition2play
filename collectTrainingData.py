import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

gestureMeaning = ["Q", "W", "E", "R", "D", "F"]
numberOfImage = 15

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

data = []
label = []

with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for meaning in gestureMeaning:
        for i in range(1, numberOfImage + 1):
            filename = f"{meaning}({i}).jpg"
            image = cv2.imread("img/" + filename)
            # cv2.imshow(filename, image)
            # cv2.waitKey(0)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = hands.process(image)

            if not results.multi_hand_landmarks:
                print("not recognised:", filename)
                continue

            # annotate by landmarks
            annotated_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            for landmark in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    landmark,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            # cv2.imshow(filename, annotated_image)
            # cv2.waitKey(0)  # check annotated images

            # summarise landmarks
            temp_data = []
            for landmark in results.multi_hand_landmarks[0].landmark:  # only 1 hand, so [0]
                temp_data.append([landmark.x, landmark.y, landmark.z])
            temp_data = np.array(temp_data)
            temp_data = temp_data - temp_data[0, :]  # relative coordinate (with respect to wrist)
            temp_data = temp_data.flatten()
            # print(temp_data)

            data.append(temp_data.tolist())
            label.append(meaning)

print(np.array(data).shape)
df = pd.DataFrame(data)  # data frame with raw data: label and relative coordinates to wrist
df.insert(0, "label", label)
print(df)

df.to_csv("gestureRawData.csv", index=False)
