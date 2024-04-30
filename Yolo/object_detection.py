import cv2
from ultralytics import YOLO

model = YOLO("yolov8x-worldv2.pt")

# Assuming you have a function to calculate distance between two points
def calculate_distance(point1, point2):
    # Calculate distance between two points (e.g., Euclidean distance)
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def navigate_based_on_objects(person_box, object_boxes, threshold_distance=100):
    closest_object_distance = float('inf')
    closest_object_direction = None

    # Calculate distances between person and nearby objects
    for object_box in object_boxes:
        object_center = ((object_box.xyxy[0][0] + object_box.xyxy[0][2]) / 2, (object_box.xyxy[0][1] + object_box.xyxy[0][3]) / 2)
        distance = calculate_distance(person_center, object_center)

        if distance < closest_object_distance:
            closest_object_distance = distance
            closest_object_direction = "right" if object_center[0] > person_center[0] else "left"

    # Provide navigation instructions based on the closest object
    if closest_object_distance < threshold_distance:
        return f"Turn {closest_object_direction} to avoid collision"
    else:
        return "Safe to continue"

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


cap = cv2.VideoCapture("http://192.168.109.246:4747/video")

while True:
    ret, frame = cap.read()  # Read a frame from the video source
    if not ret:
        break

    result_img, results = predict_and_detect(model, frame, classes=[], conf=0.5)

    # Find person and nearby objects
    person_box = None
    object_boxes = []
    for result in results:
        for box in result.boxes:
            if result.names[int(box.cls[0])] == "person":
                person_box = box
                person_center = ((box.xyxy[0][0] + box.xyxy[0][2]) / 2, (box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            else:
                object_boxes.append(box)

    # If person and objects are detected, provide navigation instructions
    if person_box and object_boxes:
        navigation_feedback = navigate_based_on_objects(person_box, object_boxes)
        cv2.putText(result_img, navigation_feedback, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", result_img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()



