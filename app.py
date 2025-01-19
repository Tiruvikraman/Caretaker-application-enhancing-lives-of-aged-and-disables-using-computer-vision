import cv2
from flask import Flask, Response, render_template
from ultralytics import YOLO

app = Flask(__name__)

# Initialize YOLO models

model1 = YOLO(r'C:\Users\tiruv\Desktop\hack2tech\Yolo\fall.pt')
model3 = YOLO("yolov8x-worldv2.pt")
print("hello")

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)
    return results


def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1,
                       font_scale=1.5):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                          (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            # Display text on top of bounding box
            class_name = result.names[int(box.cls[0])]
            confidence = float(box.conf)
            text = f"{class_name}: {confidence:.2f}"
            cv2.putText(img, text,
                        (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                        cv2.FONT_HERSHEY_PLAIN, font_scale, (255, 0, 0), text_thickness)

    return img, results


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_img1, _ = predict_and_detect(model1, frame, classes=[], conf=0.5)
        result_img3, results3 = predict_and_detect(model3, result_img1, classes=[], conf=0.5)

        collision_detected = False
        fall_detected = False

        for result in results3:
            for box in result.boxes:
                if result.names[int(box.cls[0])] == "person":
                    person_box = box
                    person_center = ((box.xyxy[0][0] + box.xyxy[0][2]) / 2, (box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                elif result.names[int(box.cls[0])] == "fall":
                    fall_detected = True
                    break
                else:
                    object_box = box

                    # Calculate distances between person and nearby objects
                    object_center = ((object_box.xyxy[0][0] + object_box.xyxy[0][2]) / 2,
                                     (object_box.xyxy[0][1] + object_box.xyxy[0][3]) / 2)
                    distance = ((person_center[0] - object_center[0]) ** 2 + (
                                person_center[1] - object_center[1]) * 2) * 0.5

                    if distance < 100:  # Modify this threshold as needed
                        collision_detected = True
                        break

        if collision_detected or fall_detected:
            yield "data: Alert!! Collision or Fall detected!!!\n\n"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/alert_feed')
def alert_feed():
    return Response(generate_frames(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True, port=8000) 
