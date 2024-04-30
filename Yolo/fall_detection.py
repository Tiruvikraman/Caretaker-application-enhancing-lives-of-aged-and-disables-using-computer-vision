import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO(r'C:\Users\tiruv\Downloads\hack3tech\Fall_Detection\fall.pt')

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
video_path = r'C:\Users\tiruv\Downloads\hack3tech\Fall-Detection\Output_videos\fall.mp4'  # Modify this with your video file path
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()  # Read a frame from the video source
    if not ret:
        break

    result_img, _ = predict_and_detect(model, frame, classes=[], conf=0.5)

    # Display the resulting frame
    cv2.imshow("Frame", result_img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
