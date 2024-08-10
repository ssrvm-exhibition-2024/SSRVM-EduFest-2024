import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from gtts import gTTS
from playsound import playsound

video = cv2.VideoCapture(0)

objects = []

while True:

    ret, frame = video.read()
    frame = cv2.flip(frame,1)
    bbox, object, conf = cv.detect_common_objects(frame)
    output_image = draw_bbox(frame, bbox, object, conf)

    cv2.imshow("Object Detection", output_image)

    for item in object:
        if item in objects:
            pass
        else:
            objects.append(item)

    if cv2.waitKey(1) == ord(" "):
        break

i = 0
new_sentence = []

for object in objects:
    if i == 0:
        new_sentence.append(f"I saw a {object}")
    else:
        new_sentence.append(f", a {object}")

    i += 1

output_sentence = " ".join(new_sentence)
print(output_sentence)

tts = gTTS(text=output_sentence, lang='en')
tts.save("detected_objects.mp3")
playsound('detected_objects.mp3')

cv2.destroyAllWindows()