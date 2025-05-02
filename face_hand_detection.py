import cv2
import mediapipe as mp

# Initialize Mediapipe modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

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

        # Draw hand landmarks and check distance
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

                # Get the z-coordinate of the index finger tip
                z_coordinate = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                # Determine if the hand is near or far (threshold can be adjusted)
                if z_coordinate < -0.1:  # Negative z means the hand is close to the camera
                    print("Hand is near the camera")
                else:
                    print("Hand is not near the camera")

        # Display the frame
        cv2.imshow("Hand Detection with Distance", frame)

        # Break the loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
