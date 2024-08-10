import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from gtts import gTTS
from playsound import playsound
import os

# Initialize video capture from the default camera
video = cv2.VideoCapture(0)

# Initialize a set to store detected objects (avoids duplicates)
detected_objects = set()

while True:
    # Capture a frame from the video feed
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Flip the frame horizontally for a mirrored effect
    frame = cv2.flip(frame, 1)

    # Detect common objects in the frame
    bbox, objects, conf = cv.detect_common_objects(frame)

    # Draw bounding boxes around detected objects
    output_image = draw_bbox(frame, bbox, objects, conf)

    # Display the output image with bounding boxes
    cv2.imshow("Object Detection", output_image)

    # Add newly detected objects to the set
    for obj in objects:
        detected_objects.add(obj)

    # Exit the loop when the spacebar is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Construct a sentence describing the detected objects
new_sentence = []
for i, obj in enumerate(detected_objects):
    if i == 0:
        new_sentence.append(f"I saw a {obj}")
    else:
        new_sentence.append(f", a {obj}")

# Join the list into a single sentence
output_sentence = " ".join(new_sentence)
print(output_sentence)

# Convert the text sentence to speech and save it as an MP3 file
tts = gTTS(text=output_sentence, lang='en')
audio_file = "detected_objects.mp3"
tts.save(audio_file)

# Play the generated audio file
playsound(audio_file)

# Remove the temporary audio file
os.remove(audio_file)
