import cv2
import process as pr

cap = cv2.VideoCapture('yol15.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    # Process each frame
    if ret:
        pr.process(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Video bittiğinde veya okunamadığında döngüden çık
        break

cap.release()
cv2.destroyAllWindows()
