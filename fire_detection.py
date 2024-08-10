import cv2
import numpy as np
import winsound
import time

# Initialize video capture from webcam
cap = cv2.VideoCapture(0)

# Track the time of the last beep to prevent frequent beeping
last_beep_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the current frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define HSV color range for detecting fire
    lower_fire = np.array([0, 100, 100])
    upper_fire = np.array([20, 255, 255])  # Adjusted for broader color range

    # Create a binary mask where fire colors are within the defined range
    mask = cv2.inRange(hsv, lower_fire, upper_fire)

    # Apply erosion and dilation to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours of detected fire regions in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fire_detected = False
    # Draw bounding rectangles around large contours (potential fire regions)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Filter out small contours
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            fire_detected = True

    # Beep to alert when fire is detected
    if fire_detected:
        current_time = time.time()
        if current_time - last_beep_time > 5:  # Beep once every 5 seconds
            winsound.Beep(1000, 500)  # 1000 Hz, 0.5 seconds
            last_beep_time = current_time

    # Display the processed frame
    cv2.imshow('Fire Detection', frame)

    # Exit the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
