import cv2
import mediapipe as mp

# Initialize Mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the pre-trained MobileNet SSD model for detecting obstacles
model_path = "MobileNetSSD_deploy.caffemodel"
config_path = "MobileNetSSD_deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_path, model_path)

# Define the class labels for the SSD model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Open webcam feed
cap = cv2.VideoCapture(0)  # 0 selects the default webcam

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to RGB (Mediapipe requires RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Hand detection
        hand_results = hands.process(rgb_frame)

        # Detect hands and label them
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

        # Prepare the frame for MobileNet SSD
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # Loop through detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Threshold to filter weak detections
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]

                # Skip hands since they are detected by Mediapipe
                if label == "person":
                    continue

                # Get bounding box for the obstacle
                box = detections[0, 0, i, 3:7] * \
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                (startX, startY, endX, endY) = box.astype("int")

                # Draw bounding box and label
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                cv2.putText(frame, "Obstacle", (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Hand and Obstacle Detection", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
