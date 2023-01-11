import cv2
import time

gestureMeaning = ["Q", "W", "E", "R", "D", "F", "Do Nothing"]
numberOfImage = 15

cap = cv2.VideoCapture(0)

gestureCounter = 0
number = 1

while 1:
    if gestureCounter >= len(gestureMeaning):
        break

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)
    filename = gestureMeaning[gestureCounter] + f"({number})"
    cv2.imshow(filename, frame)

    key = cv2.waitKey(1)
    if key == 115:  # press s key to save
        cv2.imwrite(f"img/{filename}.jpg", frame)
        cv2.destroyWindow(filename)
        number += 1
        if number > numberOfImage:
            gestureCounter += 1
            number = 1
    elif key == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
