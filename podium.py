import cv2
import mediapipe as visuals
import numpy as np

pose = visuals.solutions.pose
marker = visuals.solutions.drawing_utils
pose_detect = pose.Pose(
    static_image_mode = False,
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cam = cv2.VideoCapture(1)
while cam.isOpened():
    success, frame = cam.read()
    if not success :
        print("Image Detection Failed!")
        continue

    h ,w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_detect.process(rgb_frame)

    if results.pose_world_landmarks :
        world_landmarks = results.pose_world_landmarks.landmark
        landmarks = results.pose_landmarks.landmark
        try :
            l_shoulder_x_tracer = landmarks[pose.PoseLandmark.LEFT_SHOULDER].x * w
            l_shoulder_y_tracer = landmarks[pose.PoseLandmark.LEFT_SHOULDER].y * h
            l_shoulder_x = int(l_shoulder_x_tracer)
            l_shoulder_y = int(l_shoulder_y_tracer)
            l_shoulder = np.array([l_shoulder_x_tracer, l_shoulder_y_tracer])

            r_shoulder_x_tracer = landmarks[pose.PoseLandmark.RIGHT_SHOULDER].x * w
            r_shoulder_y_tracer = landmarks[pose.PoseLandmark.RIGHT_SHOULDER].y * h
            r_shoulder_x = int(r_shoulder_x_tracer)
            r_shoulder_y = int(r_shoulder_y_tracer)
            r_shoulder = np.array([r_shoulder_x_tracer, r_shoulder_y_tracer])

            l_hip_x_tracer = landmarks[pose.PoseLandmark.LEFT_HIP].x * w
            l_hip_y_tracer = landmarks[pose.PoseLandmark.LEFT_HIP].y * h
            l_hip_x = int(l_hip_x_tracer)
            l_hip_y = int(l_hip_y_tracer)
            l_hip = np.array([l_hip_x_tracer, l_hip_y_tracer])

            r_hip_x_tracer = landmarks[pose.PoseLandmark.RIGHT_HIP].x * w
            r_hip_y_tracer = landmarks[pose.PoseLandmark.RIGHT_HIP].y * h
            r_hip_x = int(r_hip_x_tracer)
            r_hip_y = int(r_hip_y_tracer)
            r_hip = np.array([r_hip_x_tracer, r_hip_y_tracer])


            mid_shoulder = (l_shoulder + r_shoulder) / 2.0
            mid_hip = (l_hip + r_hip) / 2.0
            shoulder_to_hip_distance = np.linalg.norm(mid_shoulder - mid_hip)
        
            t_shoulder = (int(mid_shoulder[0]) , int(mid_shoulder[1]))
            t_hip = (int(mid_hip[0]), int(mid_hip[1]))

            cv2.circle(frame, (t_shoulder[0], t_shoulder[1]), 8, (0,200,255), -1)
            cv2.circle(frame, (t_hip[0], t_hip[1]), 8, (0,200,255), -1)
            cv2.circle(frame, (l_shoulder_x, l_shoulder_y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (r_shoulder_x, r_shoulder_y), 8, (0, 255, 0), -1)
            cv2.circle(frame, (l_hip_x, l_hip_y), 8, (0,0,255), -1)
            cv2.circle(frame, (r_hip_x, r_hip_y), 8, (0,0,255), -1)
            
            ht_str = f"Approx Height : {shoulder_to_hip_distance:.1f} px"
            str_position = (t_shoulder[0] + 15, t_shoulder[1] + 15)
            cv2.putText(frame, ht_str, str_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
            cv2.line(frame, (l_shoulder_x, l_shoulder_y), (r_shoulder_x, r_shoulder_y), (255,255,0), 2)
            cv2.line(frame, (l_hip_x, l_hip_y), (r_hip_x, r_hip_y), (255,255,0), 2)
            cv2.line(frame, (t_shoulder[0], t_shoulder[1]), (t_hip[0], t_hip[1]), (255,255,0), 2)
            cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_hip_x, l_hip_y), (255,255,0), 1)
            cv2.line(frame, (r_shoulder_x, r_shoulder_y), (r_hip_x, r_hip_y), (255,255,0), 1)

        except IndexError:
            pass

    cv2.imshow('HeightCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
pose_detect.close()