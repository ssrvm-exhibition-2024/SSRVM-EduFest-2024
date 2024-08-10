import cv2
import numpy as np
import winsound

# Load the video stream (replace 'video.mp4' with your video file or '0' for webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of fire colors in HSV
    lower_fire = np.array([0, 100, 100])
    upper_fire = np.array([20, 200, 200])

    # Threshold the HSV image to get only fire colors
    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours of fire regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Iterate through contours and draw rectangles around fire regions
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # adjust this value to filter out small regions
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Play a beep sound when fire is detected
            winsound.Beep(344, 1000)  # 2500 Hz frequency, 1 second duration

    # Display the output
    cv2.imshow('Fire Detection', frame)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

