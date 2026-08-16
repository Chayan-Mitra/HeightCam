import cv2
import mediapipe as visuals
import numpy as np
from utils import FaceGateWay

def main():
    instigator = FaceGateWay()

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
        person_active = instigator.person_is_present(frame)
        #if person_active:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_detect.process(rgb_frame)

        if results.pose_world_landmarks :
            world_landmarks = results.pose_world_landmarks.landmark
            landmarks = results.pose_landmarks.landmark
            try :
                l_shoulder_3D = np.array([world_landmarks[pose.PoseLandmark.LEFT_SHOULDER].x, world_landmarks[pose.PoseLandmark.LEFT_SHOULDER].y, world_landmarks[pose.PoseLandmark.LEFT_SHOULDER].z])
                l_shoulder_x_tracer = landmarks[pose.PoseLandmark.LEFT_SHOULDER].x * w
                l_shoulder_y_tracer = landmarks[pose.PoseLandmark.LEFT_SHOULDER].y * h
                l_shoulder_x = int(l_shoulder_x_tracer)
                l_shoulder_y = int(l_shoulder_y_tracer)
                l_shoulder = np.array([l_shoulder_x_tracer, l_shoulder_y_tracer])

                r_shoulder_3D = np.array([world_landmarks[pose.PoseLandmark.RIGHT_SHOULDER].x,world_landmarks[pose.PoseLandmark.RIGHT_SHOULDER].y, world_landmarks[pose.PoseLandmark.RIGHT_SHOULDER].z])
                r_shoulder_x_tracer = landmarks[pose.PoseLandmark.RIGHT_SHOULDER].x * w
                r_shoulder_y_tracer = landmarks[pose.PoseLandmark.RIGHT_SHOULDER].y * h
                r_shoulder_x = int(r_shoulder_x_tracer)
                r_shoulder_y = int(r_shoulder_y_tracer)
                r_shoulder = np.array([r_shoulder_x_tracer, r_shoulder_y_tracer])

                l_hip_3D = np.array([world_landmarks[pose.PoseLandmark.LEFT_HIP].x, world_landmarks[pose.PoseLandmark.LEFT_HIP].y, world_landmarks[pose.PoseLandmark.LEFT_HIP].z])
                l_hip_x_tracer = landmarks[pose.PoseLandmark.LEFT_HIP].x * w
                l_hip_y_tracer = landmarks[pose.PoseLandmark.LEFT_HIP].y * h
                l_hip_x = int(l_hip_x_tracer)
                l_hip_y = int(l_hip_y_tracer)
                l_hip = np.array([l_hip_x_tracer, l_hip_y_tracer])

                r_hip_3D = np.array([world_landmarks[pose.PoseLandmark.RIGHT_HIP].x, world_landmarks[pose.PoseLandmark.RIGHT_HIP].y, world_landmarks[pose.PoseLandmark.RIGHT_HIP].z])
                r_hip_x_tracer = landmarks[pose.PoseLandmark.RIGHT_HIP].x * w
                r_hip_y_tracer = landmarks[pose.PoseLandmark.RIGHT_HIP].y * h
                r_hip_x = int(r_hip_x_tracer)
                r_hip_y = int(r_hip_y_tracer)
                r_hip = np.array([r_hip_x_tracer, r_hip_y_tracer])

                l_knee_3D = np.array([world_landmarks[pose.PoseLandmark.LEFT_KNEE].x, world_landmarks[pose.PoseLandmark.LEFT_KNEE].y, world_landmarks[pose.PoseLandmark.LEFT_KNEE].z])
                l_knee_x_tracer = landmarks[pose.PoseLandmark.LEFT_KNEE].x * w
                l_knee_y_tracer = landmarks[pose.PoseLandmark.LEFT_KNEE].y * h
                l_knee_x = int(l_knee_x_tracer)
                l_knee_y = int(l_knee_y_tracer)
                l_knee = np.array([l_knee_x, l_knee_y])

                r_knee_3D = np.array([world_landmarks[pose.PoseLandmark.RIGHT_KNEE].x, world_landmarks[pose.PoseLandmark.RIGHT_KNEE].y, world_landmarks[pose.PoseLandmark.RIGHT_KNEE].z])
                r_knee_x_tracer = landmarks[pose.PoseLandmark.RIGHT_KNEE].x * w
                r_knee_y_tracer = landmarks[pose.PoseLandmark.RIGHT_KNEE].y * h
                r_knee_x = int(r_knee_x_tracer)
                r_knee_y = int(r_knee_y_tracer)
                r_knee = np.array([r_knee_x, r_knee_y])

                r_ankel_3D = np.array([world_landmarks[pose.PoseLandmark.RIGHT_ANKLE].x, world_landmarks[pose.PoseLandmark.RIGHT_ANKLE].y, world_landmarks[pose.PoseLandmark.RIGHT_ANKLE].z])
                r_ankel_x_tracer = landmarks[pose.PoseLandmark.RIGHT_ANKLE].x * w
                r_ankel_y_tracer = landmarks[pose.PoseLandmark.RIGHT_ANKLE].y * h
                r_ankel_x = int(r_ankel_x_tracer)
                r_ankel_y = int(r_ankel_y_tracer)
                r_ankel = np.array([r_ankel_x, r_ankel_y])

                l_ankel_3D = np.array([world_landmarks[pose.PoseLandmark.LEFT_ANKLE].x, world_landmarks[pose.PoseLandmark.LEFT_ANKLE].y, world_landmarks[pose.PoseLandmark.LEFT_ANKLE].z])
                l_ankel_x_tracer = landmarks[pose.PoseLandmark.LEFT_ANKLE].x * w
                l_ankel_y_tracer = landmarks[pose.PoseLandmark.LEFT_ANKLE].y * h
                l_ankel_x = int(l_ankel_x_tracer)
                l_ankel_y = int(l_ankel_y_tracer)
                l_ankel = np.array([l_ankel_x, l_knee_y])

                cv2.circle(frame, (l_knee_x, l_knee_y), 8, (0, 90,255), -1)
                cv2.circle(frame, (l_ankel_x, l_ankel_y), 8, (0,90,255), -1)
                cv2.circle(frame, (r_knee_x, r_knee_y), 8, (0,90,255), -1)
                cv2.circle(frame, (r_ankel_x, r_ankel_y), 8, (0,90,255), -1)

                mid_shoulder_3D = (l_shoulder_3D + r_shoulder_3D) / 2.0
                mid_hip_3D = (l_hip_3D + r_hip_3D) / 2.0
                mid_ankel_3D = (l_ankel_3D + r_ankel_3D) / 2.0
                mid_knee_3D = (l_knee_3D + r_knee_3D) / 2.0
                hip_to_knee_distance_3D = np.linalg.norm(mid_hip_3D - mid_knee_3D)
                knee_to_ankel_distance_3D = np.linalg.norm(mid_knee_3D - mid_ankel_3D)
                shoulder_to_hip_distance_3D = np.linalg.norm(mid_shoulder_3D - mid_hip_3D)
                hip_shoulder_cm_ratio = shoulder_to_hip_distance_3D * 100.0
                hip_knee_cm_ratio = hip_to_knee_distance_3D * 100.0
                knee_ankel_cm_ratio = knee_to_ankel_distance_3D * 100.0
                mid_shoulder = (l_shoulder + r_shoulder) / 2.0
                mid_hip = (l_hip + r_hip) / 2.0
                mid_knee = (l_knee + r_knee) / 2.0
                mid_ankel = (l_ankel + r_ankel) / 2.0
                shoulder_to_hip_2D_overlay = np.linalg.norm(mid_shoulder - mid_hip)
                hip_to_knee_2D_overlay = np.linalg.norm(mid_hip - mid_knee)
                knee_to_ankel_2D_overlay = np.linalg.norm(mid_knee - mid_ankel)
                t_shoulder = (int(mid_shoulder[0]) , int(mid_shoulder[1]))
                t_hip = (int(mid_hip[0]), int(mid_hip[1]))
                t_knee = (int(mid_knee[0]), int(mid_knee[1]))
                #t_ankel = (int(mid_ankel[0]), int(mid_ankel[1]))
                cv2.circle(frame, (t_shoulder[0], t_shoulder[1]), 8, (0,200,255), -1)
                cv2.circle(frame, (t_hip[0], t_hip[1]), 8, (0,200,255), -1)
                cv2.circle(frame, (l_shoulder_x, l_shoulder_y), 8, (0, 255, 0), -1)
                cv2.circle(frame, (r_shoulder_x, r_shoulder_y), 8, (0, 255, 0), -1)
                cv2.circle(frame, (l_hip_x, l_hip_y), 8, (0,0,255), -1)
                cv2.circle(frame, (r_hip_x, r_hip_y), 8, (0,0,255), -1)
                height = hip_shoulder_cm_ratio + hip_knee_cm_ratio + knee_ankel_cm_ratio
                cm_str = f"Approx Height (in cm) : {height:.1f} cm"
                sh_str = f"Approx Shoulder to Hip length : {shoulder_to_hip_2D_overlay:.1f} px"
                hk_str = f"Approx Hip to Knee length : {hip_to_knee_2D_overlay:.1f} px"
                ka_str = f"Approx Knee to Ankle length : {knee_to_ankel_2D_overlay:.1f} px"
                str_position = (t_shoulder[0] + 15, t_shoulder[1] + 15)
                cv2.putText(frame, sh_str, str_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
                cv2.putText(frame, hk_str, (t_hip[0] + 15, t_hip[0] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)
                cv2.putText(frame, ka_str, (t_knee[0] + 15, t_knee[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,100,255), 2)
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (r_shoulder_x, r_shoulder_y), (255,255,0), 2)
                cv2.line(frame, (l_hip_x, l_hip_y), (r_hip_x, r_hip_y), (255,255,0), 2)
                cv2.line(frame, (t_shoulder[0], t_shoulder[1]), (t_hip[0], t_hip[1]), (255,255,0), 2)
                cv2.line(frame, (l_shoulder_x, l_shoulder_y), (l_hip_x, l_hip_y), (255,255,0), 1)
                cv2.line(frame, (r_shoulder_x, r_shoulder_y), (r_hip_x, r_hip_y), (255,255,0), 1)
                cv2.line(frame, (r_hip_x, r_hip_y), (r_knee_x, r_knee_y), (255,255,0), 2)
                cv2.line(frame, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), (255,255,0), 2)
                cv2.line(frame, (r_knee_x, r_knee_y), (r_ankel_x, r_ankel_y), (255,255,0), 2)
                cv2.line(frame, (l_knee_x, l_knee_y), (l_ankel_x, l_ankel_y), (255,255,0), 2)
                cv2.putText(frame, cm_str, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,100,255), 2)

            except IndexError:
                pass
        # else :
        #     status_update = "STANDBY..."
        #     cv2.putText(frame, status_update, (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, lineType=cv2.LINE_AA)


        cv2.imshow('HeightCam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    pose_detect.close()

if __name__ == '__main__':
    main()