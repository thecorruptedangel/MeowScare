import cv2
import numpy as np
from ultralytics import YOLO

# Flag to choose between CPU and GPU
use_gpu = True

# Load the YOLOv8s model
if use_gpu:
    model = YOLO('yolov8s.pt').cuda()
else:
    model = YOLO('yolov8s.pt')
model.verbose = False  # Turn off verbosity

# Open the video capture
cap = cv2.VideoCapture('video.mp4')

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Run the YOLOv8s model on the frame
    results = model(frame, verbose=False)[0]
    
    # Loop through the detected objects
    for box in results.boxes:
        # Check if the detected object is a cat or a dog
        if int(box.cls[0]) in [16, 17]:  # Cat (16) and Dog (17) class IDs in COCO dataset
            # Get the bounding box coordinates
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            
            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Print the class and coordinates of the detected object
            if int(box.cls[0]) == 16:
                print(f'Cat detected at ({x1}, {y1}, {x2}, {y2})')
            else:
                print(f'Dog detected at ({x1}, {y1}, {x2}, {y2})')
    
    # Display the frame with bounding boxes
    cv2.imshow('Animal Detection', frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
cap.release()
cv2.destroyAllWindows()