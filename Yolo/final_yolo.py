import cv2
from ultralytics import YOLO
import time
# Initialize YOLO models
model1 = YOLO(r'Yolo/fall.pt')
model3 = YOLO("yolov8x-worldv2.pt")



# Adjust the scale factor to downscale frames
scale_factor = 0.5

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

# Open the video file
video_path = r'C:\Users\tiruv\Downloads\hack3tech\Fall_Detection_Using_Yolov8\fall_detection_1 (1).mp4'  # Modify this with your video file path
cap = cv2.VideoCapture(video_path)

frame_skip = 2  # Skip every 2 frames
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()  # Read a frame from the video source
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames

    # Downscale frame
    frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)

    # Process frame with first YOLO model
    result_img1, _ = predict_and_detect(model1, frame, classes=[], conf=0.5)

    # Process frame with third YOLO model
    result_img3, results3 = predict_and_detect(model3, result_img1, classes=[], conf=0.5)

    # Find person and nearby objects
    person_box = None
    object_boxes = []
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

        # Provide navigation instructions based on the closest object
        if closest_object_distance < 100:
            navigation_feedback = f"Turn {closest_object_direction} to avoid collision"
        else:
            navigation_feedback = "Safe to continue"

        cv2.putText(result_img3, navigation_feedback, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Frame", result_img3)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print processing time
end_time = time.time()
processing_time = end_time - start_time
print("Processing time:", processing_time, "seconds")

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
