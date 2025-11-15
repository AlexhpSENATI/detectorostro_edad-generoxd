from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from src.utils import load_models
from src.age_gender import AGE_LIST, GENDER_LIST

app = FastAPI()

# -------------------
# Cargar los modelos
# -------------------
faceNet, ageNet, genderNet = load_models()


@app.get("/")
def root():
    return {"message": "API Age-Gender funcionando!"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Leer contenido en bytes
    contents = await file.read()

    # Convertir a imagen OpenCV
    np_img = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "No se pudo leer la imagen"}

    h, w = frame.shape[:2]

    # Preparar blob para detección de rostros
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0)
    )
    faceNet.setInput(blob)
    detections = faceNet.forward()

    results = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.6:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = box.astype(int)

        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # Preparar blob para edad/género
        blob2 = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False
        )

        # GÉNERO
        genderNet.setInput(blob2)
        gender = GENDER_LIST[genderNet.forward()[0].argmax()]

        # EDAD
        ageNet.setInput(blob2)
        age = AGE_LIST[ageNet.forward()[0].argmax()]

        results.append({
            "gender": gender,
            "age": age,
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return {"results": results}
