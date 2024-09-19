import cv2
import dlib
import pyautogui
from scipy.spatial import distance as dist
import numpy as np


# Initialize the dlib face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Eye aspect ratio to indicate blink
EYE_AR_THRESH = 0.2
EYE_AR_CONSEC_FRAMES = 3

# Counters
blink_counter = 0
double_blink_counter = 0
blink_detected = False

# Calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Detect blinks and take action
def detect_blink_and_take_action():
    global blink_counter, double_blink_counter, blink_detected

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret or frame is None:
            print("Failed to grab frame")
            break

        # Ensure the image is in 8-bit grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Debugging: Check the image type and shape
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
        print(f"Gray shape: {gray.shape}, dtype: {gray.dtype}")

        # Ensure the grayscale image is of type uint8
        if gray.dtype != 'uint8':
            print("Gray image is not 8-bit. Exiting.")
            break

        # Detect faces
        try:
            rects = detector(gray)
        except Exception as e:
            print(f"Error during face detection: {e}")
            break

        for rect in rects:
            shape = predictor(gray, rect)
            shape = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            # Extract the left and right eye coordinates
            leftEye = shape[36:42]
            rightEye = shape[42:48]

            # Convert to numpy arrays for easier processing
            leftEyePts = np.array(leftEye)
            rightEyePts = np.array(rightEye)

            # Calculate the eye aspect ratio for both eyes
            leftEAR = eye_aspect_ratio(leftEyePts)
            rightEAR = eye_aspect_ratio(rightEyePts)

            # Average the EAR values to get a single EAR value
            ear = (leftEAR + rightEAR) / 2.0

            # Check if the eye aspect ratio is below the blink threshold
            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                # If the eyes were closed for a sufficient number of frames
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    # A blink is detected
                    blink_detected = True
                    double_blink_counter += 1

                    # If a double blink is detected
                    if double_blink_counter == 2:
                        # Trigger action (e.g., open a file or folder)
                        print("Double blink detected! Taking action.")
                        pyautogui.hotkey('win', 'e')  # This opens the file explorer on Windows

                        # Reset the double blink counter after action
                        double_blink_counter = 0

                # Reset the blink counter and blink_detected flag
                blink_counter = 0
                blink_detected = False

        # Show the frame
        cv2.imshow("Blink Detection", frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the blink detection and take action
detect_blink_and_take_action()

