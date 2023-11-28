# encoding:utf-8
"""
__time__    : 2023/11/25 20:52
__author__  : LSY
"""
import sys
import time
import math
import cv2

from FallDownDetection.model.src_import import *


def toggle_fullscreen(window_name):
    current_state = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
    if current_state == cv2.WINDOW_NORMAL:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)


if __name__ == '__main__':

    # 初始化一些变量
    show_contour_flag = False

    # Global variables to calculate FPS
    COUNTER, FPS = 0, 0
    START_TIME = time.time()
    DETECTION_RESULT = None

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 创建窗口并显示图像
    window_name = "mainAPP_FallDownDete"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, 0, 0)
    cv2.resizeWindow(window_name, (1080, 720))

    # Visualization parameters
    row_size = 50  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 0)  # black
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10
    overlay_alpha = 0.5
    mask_color = (100, 100, 0)  # cyan

    def save_result(result: vision.PoseLandmarkerResult,
                    unused_output_image: mp.Image, timestamp_ms: int):
        global FPS, COUNTER, START_TIME, DETECTION_RESULT

        # Calculate the FPS
        if COUNTER % fps_avg_frame_count == 0:
            FPS = fps_avg_frame_count / (time.time() - START_TIME)
            START_TIME = time.time()

        DETECTION_RESULT = result
        COUNTER += 1

    # Initialize the pose landmarker model
    base_options = python.BaseOptions(model_asset_path='model/human_gesture.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        result_callback=save_result)
    detector = vision.PoseLandmarker.create_from_options(options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        # 初始化动作标签
        action_label = "Unknown"

        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        image = cv2.flip(image, 1)

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Run pose landmarker using the model.
        detector.detect_async(mp_image, time.time_ns() // 1_000_000)

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(FPS)
        text_location = (left_margin, row_size)
        current_frame = image
        cv2.putText(current_frame, fps_text, text_location,
                    cv2.FONT_HERSHEY_DUPLEX,
                    font_size, text_color, font_thickness, cv2.LINE_AA)

        if DETECTION_RESULT:
            # Draw landmarks.
            for pose_landmarks in DETECTION_RESULT.pose_landmarks:
                # Draw the pose landmarks.
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y,
                                                    z=landmark.z) for landmark
                    in pose_landmarks
                ])
                if show_contour_flag:
                    mp_drawing.draw_landmarks(
                        current_frame,
                        pose_landmarks_proto,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing_styles.get_default_pose_landmarks_style(),
                        # mp_drawing.DrawingSpec(thickness=20, circle_radius=4)
                        mp_drawing.DrawingSpec(color=(0, 255, 124), thickness=2, circle_radius=4)
                    )

                # 获取关键点坐标
                left_hip = pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                left_knee = pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
                right_hip = pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                right_knee = pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
                shoulder = pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

                # 计算左右腿的角度
                left_leg_angle = abs(left_hip.y - left_knee.y)
                right_leg_angle = abs(right_hip.y - right_knee.y)

                # 计算身体的倾斜角度
                body_tilt_angle = abs(math.degrees(math.atan2(shoulder.y - left_hip.y, shoulder.x - left_hip.x)))

                # 设定动作阈值
                sit_threshold = 0.1  # 这是一个示例值，你可能需要根据实际情况调整
                fall_threshold = 50  # 这是一个示例值，你可能需要根据实际情况调整

                # print("sit_threshold: ", sit_threshold)
                # print("body_tilt_angle: ", body_tilt_angle)
                # 判断动作
                if left_leg_angle < sit_threshold or right_leg_angle < sit_threshold:
                    action_label = "Sitting"
                elif body_tilt_angle < fall_threshold or body_tilt_angle > 180 - fall_threshold:
                    action_label = "Fallen"
                else:
                    action_label = "Standing"

        cv2.putText(current_frame, action_label, (int(current_frame.shape[0]/2-10), 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4, cv2.LINE_AA)

        cv2.imshow('mainAPP_FallDownDete', current_frame)

        cv2.putText(current_frame, "Press 'f' to toggle fullscreen", (10, current_frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 155), 2, cv2.LINE_AA)
        cv2.putText(current_frame, "Press 'r' to toggle contour", (10, current_frame.shape[0] - 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 155), 2, cv2.LINE_AA)
        cv2.putText(current_frame, "Press 'ESC/q' to quit", (10, current_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 155), 2, cv2.LINE_AA)
        cv2.imshow(window_name, current_frame)

        key = cv2.waitKey(1) & 0xFF
        # 按下 'f' 键切换全屏和退出全屏
        if key in [ord('f'), ord('F')]:
            toggle_fullscreen(window_name)
        # 按下 'r' 键切换轮廓显示
        elif key in [ord('r'), ord('R')]:
            show_contour_flag = not show_contour_flag
        # 按下 'Esc' 键或 'q' 键退出全屏
        elif key in [27, ord('q'), ord('Q')]:
            break
        # 检测窗口是否关闭
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

