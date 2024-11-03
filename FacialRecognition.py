import cv2
import numpy as np
from deepface import DeepFace

# Load
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
cap = cv2.VideoCapture(0)

is_smiling = False

class Stabilizer:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.positions = []

    def smooth(self, position):
        self.positions.append(position)
        if len(self.positions) > self.window_size:
            self.positions.pop(0)
        avg_position = np.mean(self.positions, axis=0).astype(int)
        return avg_position

stabilizer = Stabilizer(window_size=10)

def detect_faces(gray_frame):
    front_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    profile_faces = profile_face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return list(front_faces) + list(profile_faces)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    all_faces = detect_faces(gray)

    for (x, y, w, h) in all_faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=10, minSize=(25, 25))
        smoothed_position = stabilizer.smooth((x, y, w, h))
        x, y, w, h = map(int, smoothed_position)

        face_img = frame[y:y + h, x:x + w]
        try:
            analysis = DeepFace.analyze(face_img, actions=['emotion'], enforce_detection=False)
            emotion_label = analysis['dominant_emotion']
        except Exception as e:
            emotion_label = "Unknown"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        if len(smiles) > 0:
            if not is_smiling:
                print("Smile detected!")
                is_smiling = True
            cv2.putText(frame, "Smiling!", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            if is_smiling:
                print("Smile not detected!")
                is_smiling = False
            cv2.putText(frame, "Not Smiling", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
