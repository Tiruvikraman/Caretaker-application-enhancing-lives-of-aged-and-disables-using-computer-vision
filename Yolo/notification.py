import cv2
from ultralytics import YOLO
from flask import Flask, Response
import threading

app = Flask(__name__)

# Initialize YOLO models
model1 = YOLO(r'/Users/rosh/Downloads/fall.pt')
model3 = YOLO("yolov8x-worldv2.pt")

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
            # Draw bounding box
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

def send_alert():
    # Send JavaScript alert
    js_alert = "<script>alert('Collision detected!');</script>\n"
    return js_alert

def detect_collision():
    cap = cv2.VideoCapture("http://192.168.109.246:4747/video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result_img1, _ = predict_and_detect(model1, frame, classes=[], conf=0.5)
        result_img3, _ = predict_and_detect(model3, result_img1, classes=[], conf=0.5)

        # Find person and nearby objects
        person_box = None
        object_boxes = []
        collision_occurred = False
        for result in results3:
            for box in result.boxes:
                if result.names[int(box.cls[0])] == "person":
                    person_box = box
                    person_center = ((box.xyxy[0][0] + box.xyxy[0][2]) / 2, (box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                else:
                    object_boxes.append(box)

        # If person and objects are detected, provide navigation instructions
        if person_box and object_boxes:
            closest_object_distance = float('inf')
            closest_object_direction = None

            # Calculate distances between person and nearby objects
            for object_box in object_boxes:
                object_center = ((object_box.xyxy[0][0] + object_box.xyxy[0][2]) / 2, (object_box.xyxy[0][1] + object_box.xyxy[0][3]) / 2)
                distance = ((person_center[0] - object_center[0]) ** 2 + (person_center[1] - object_center[1]) ** 2) ** 0.5

                if distance < closest_object_distance:
                    closest_object_distance = distance
                    closest_object_direction = "right" if object_center[0] > person_center[0] else "left"

                # Check for collision
                if person_box.xyxy[0][0] < object_box.xyxy[0][2] and person_box.xyxy[0][2] > object_box.xyxy[0][0] \
                        and person_box.xyxy[0][1] < object_box.xyxy[0][3] and person_box.xyxy[0][3] > object_box.xyxy[0][1]:
                    collision_occurred = True

            if collision_occurred:
                yield send_alert()

@app.route('/')
def index():
    return Response(detect_collision(), mimetype='text/html')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
