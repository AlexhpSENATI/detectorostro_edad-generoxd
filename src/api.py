from fastapi import FastAPI, File, UploadFile
import cv2
from src.utils import load_models
from src.age_gender import AGE_LIST, GENDER_LIST


app = FastAPI()

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']

GENDER_LIST = ['Hombre', 'Mujer']

faceNet, ageNet, genderNet = load_models()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            x1, y1, x2, y2 = box.astype(int)
            face = frame[y1:y2, x1:x2]

            blob2 = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                          (78.4263377603, 87.7689143744, 114.895847746),
                                          swapRB=False)

            genderNet.setInput(blob2)
            gender = GENDER_LIST[genderNet.forward()[0].argmax()]

            ageNet.setInput(blob2)
            age = AGE_LIST[ageNet.forward()[0].argmax()]

            return {
                "edad": age,
                "genero": gender
            }

    return {"error": "No se detectó rostro"}
