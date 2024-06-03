import cv2
from datetime import datetime

videocapture = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def detectboundingbox(vid):

    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces


while True:

    result , video_frame = videocapture.read()
    if result is False:
        break
    cv2.imshow("My face detection project", video_frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        print("Escape hit, closing...")
        break

    faces = detectboundingbox(video_frame)
    now = datetime.now()

    if type(faces) is not tuple:
        img_name = "frame_at_time_"+str(now.hour)+"."+str(now.minute)+"."+str(now.second)+".png"
        cv2.imwrite(img_name, video_frame)




videocapture.release()
cv2.destroyAllWindows()
