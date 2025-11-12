import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
import time
import math
import random
import warnings

from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf.symbol_database')

MODEL_PATH = 'sign_language_model.pkl'
MODEL_FEATURE_NAMES = []
try:
    model = joblib.load(MODEL_PATH)
    MODEL_FEATURE_NAMES = model.feature_names_in_
    print(f"Loaded model (needs STOP/RESUME/POINT) and {len(MODEL_FEATURE_NAMES)} feature names.")
except FileNotFoundError:
    print(f"Error: Model file {MODEL_PATH} not found.")
    print("Have you run the data collector (with STOP/RESUME/POINT) and model trainer?")
    exit()
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

DEBOUNCE_FRAMES = 10
prediction_buffer = [""] * DEBOUNCE_FRAMES
stable_prediction = ""
letter_objects = []
last_added_time = 0
ADD_COOLDOWN = 1.0
left_hand_locked_on_space = False
cursor_pos = (0, 0)
grab_cursor_pos = (0, 0) 
right_hand_gesture = ""
dragged_object_index = -1
hover_start_time = None
hover_target_index = -1
HOVER_DELETE_SEC = 0.5
delete_cooldown_start = 0
DELETE_COOLDOWN_SEC = 0.5
bobbing_frequency = 2.0
bobbing_amplitude = 5

font_face_cv = cv2.FONT_HERSHEY_SIMPLEX
font_scale_pred_cv = 3
font_thickness_pred_cv = 4

font_path = "font.ttf"
font_size_sentence = 70
font_sentence_pil = None
use_pillow_font = False
try:

    font_sentence_pil = ImageFont.truetype(font_path, font_size_sentence)
    use_pillow_font = True
except IOError:
    font_scale_sent_cv = 2.0
    font_thickness_sent_cv = 3

drawing_spec_left = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
drawing_spec_right = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    current_time = time.time()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    current_left_prediction = ""
    right_hand_gesture = ""
    delete_index = -1

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            data_row_relative = []
            wrist_landmark = hand_landmarks.landmark[0]
            wrist_x, wrist_y, wrist_z = wrist_landmark.x, wrist_landmark.y, wrist_landmark.z
            for lm in hand_landmarks.landmark:
                data_row_relative.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
            max_abs_value = np.max(np.abs(data_row_relative))
            data_row_normalized = []
            if max_abs_value != 0:
                data_row_normalized = (np.array(data_row_relative) / max_abs_value).tolist()
            current_gesture = ""
            if data_row_normalized:
                try:
                    df_row = pd.DataFrame([data_row_normalized], columns=MODEL_FEATURE_NAMES)
                    prediction = model.predict(df_row)
                    current_gesture = prediction[0]
                except Exception as e:
                    current_gesture = ""

            if hand_label == 'Left':
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_left, connection_drawing_spec=drawing_spec_left)
                if current_gesture == "SPACE":
                    left_hand_locked_on_space = True; current_left_prediction = " "
                elif left_hand_locked_on_space:
                    if current_gesture != "SPACE": left_hand_locked_on_space = False
                    if current_gesture not in ["STOP", "RESUME", "POINT", "SPACE"]: current_left_prediction = current_gesture
                    else: current_left_prediction = ""
                elif not left_hand_locked_on_space:
                    if current_gesture not in ["STOP", "RESUME", "POINT"]: current_left_prediction = current_gesture
                    else: current_left_prediction = ""

            elif hand_label == 'Right':
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_drawing_spec=drawing_spec_right, connection_drawing_spec=drawing_spec_right)
                if current_gesture in ["STOP", "RESUME", "POINT"]: right_hand_gesture = current_gesture
                else: right_hand_gesture = ""
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                cursor_pos = (int(index_tip.x * w), int(index_tip.y * h))
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
                palm_center_x = int(((wrist.x + middle_mcp.x) / 2) * w)
                palm_center_y = int(((wrist.y + middle_mcp.y) / 2) * h)
                grab_cursor_pos = (palm_center_x, palm_center_y)

    if not left_hand_locked_on_space:
        prediction_buffer.append(current_left_prediction)
        if len(prediction_buffer) > DEBOUNCE_FRAMES:
            prediction_buffer.pop(0)
        is_stable = False
        if len(prediction_buffer) == DEBOUNCE_FRAMES:
            first_pred = prediction_buffer[0]
            if first_pred != "" and first_pred not in ["STOP", "RESUME", "POINT"] and all(pred == first_pred for pred in prediction_buffer):
                is_stable = True
                stable_prediction = first_pred
        if is_stable and stable_prediction != last_added_letter and current_time > last_added_time + ADD_COOLDOWN:
            new_letter = stable_prediction
            tw, th = 50, 50
            if use_pillow_font:
                 try:
                     l_bbox = font_sentence_pil.getbbox(new_letter)
                     tw = l_bbox[2] - l_bbox[0]
                     th = l_bbox[3] - l_bbox[1]
                 except AttributeError:
                      (tw_cv, th_cv), _ = cv2.getTextSize(new_letter, font_face_cv, font_scale_sent_cv, font_thickness_sent_cv)
                      tw, th = tw_cv, th_cv
                 except Exception as e:
                     print(f"Pillow font size error: {e}")
                     (tw_cv, th_cv), _ = cv2.getTextSize(new_letter, font_face_cv, font_scale_sent_cv, font_thickness_sent_cv)
                     tw, th = tw_cv, th_cv
            else:
                 (tw, th), _ = cv2.getTextSize(new_letter, font_face_cv, font_scale_sent_cv, font_thickness_sent_cv)

            start_x = random.randint(50, w - 50 - tw if w > 50 + tw else 50)
            start_y = random.randint(150, h - 50 if h > 150 + 50 else 150)
            letter_objects.append({
                'char': new_letter, 'x': start_x, 'y': start_y, 'w': tw, 'h': th,
                'start_time': current_time
            })
            last_added_letter = new_letter
            last_added_time = current_time
            stable_prediction = ""
        elif not is_stable and stable_prediction == "":
             last_added_letter = None
    else:
        stable_prediction = ""
        if last_added_letter != " ":
             last_added_letter = " "

    is_grabbing = (right_hand_gesture == "RESUME")
    is_dropping = (right_hand_gesture == "STOP")
    if dragged_object_index != -1:
        if dragged_object_index < len(letter_objects):
            obj = letter_objects[dragged_object_index]
            if is_dropping: dragged_object_index = -1
            elif is_grabbing:
                 obj['x'] = grab_cursor_pos[0] - obj['w'] // 2
                 obj['y'] = grab_cursor_pos[1] + obj['h'] // 2
            elif not is_grabbing: dragged_object_index = -1
        else: dragged_object_index = -1
    elif is_grabbing:
        for i in range(len(letter_objects) - 1, -1, -1):
            obj = letter_objects[i]
            y_offset = bobbing_amplitude * math.sin((current_time - obj['start_time']) * bobbing_frequency)
            display_y = obj['y'] + int(y_offset)
            x_offset = bobbing_amplitude * math.cos((current_time - obj['start_time']) * bobbing_frequency * 0.7 + i)
            display_x = obj['x'] + int(x_offset)
            x1, y1 = display_x, display_y - obj['h']
            x2, y2 = display_x + obj['w'], display_y
            if (grab_cursor_pos[0] > x1 and grab_cursor_pos[0] < x2) and \
               (grab_cursor_pos[1] > y1 and grab_cursor_pos[1] < y2):
                dragged_object_index = i; break

    key = cv2.waitKey(5) & 0xFF
    if key == ord('0'):
        break
    pred_box_x1, pred_box_y1 = 0, 0
    pred_box_x2, pred_box_y2 = 120, 100
    cv2.rectangle(frame, (pred_box_x1, pred_box_y1), (pred_box_x2, pred_box_y2), (0, 0, 0), -1)
    pred_display = stable_prediction if stable_prediction != " " else "_"
    if pred_display not in ["STOP", "RESUME", "POINT"]:
        cv2.putText(frame, pred_display, (20, 80), font_face_cv, font_scale_pred_cv, (0, 255, 0), font_thickness_pred_cv, cv2.LINE_AA)

    pil_image = None
    draw = None
    if use_pillow_font:
        # Convert frame *after* drawing the prediction box
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        draw = ImageDraw.Draw(pil_image)

    color_normal_pil = (0,0,0)
    color_hover_pil = (255,255,255)
    color_drag_pil = (255,255,255)

    can_delete_now = time.time() > delete_cooldown_start
    currently_hovering_index = -1

    for i in range(len(letter_objects) - 1, -1, -1):
        if i >= len(letter_objects): continue
        obj = letter_objects[i]
        letter = obj['char']
        base_x, base_y = obj['x'], obj['y']
        text_w, text_h = obj['w'], obj['h']

        x_offset = 0; y_offset = 0
        if i != dragged_object_index:
            y_offset = bobbing_amplitude * math.sin((current_time - obj['start_time']) * bobbing_frequency)
            x_offset = bobbing_amplitude * math.cos((current_time - obj['start_time']) * bobbing_frequency * 0.7 + i)
        display_x = base_x + int(x_offset)
        display_y_baseline = base_y + int(y_offset)

        if use_pillow_font:
            draw_y_pil = display_y_baseline - text_h
            x1, y1 = display_x, draw_y_pil
            x2, y2 = display_x + text_w, draw_y_pil + text_h
        else:
            x1, y1 = display_x, display_y_baseline - text_h
            x2, y2 = display_x + text_w, display_y_baseline

        is_hovering = False
        if dragged_object_index == -1 and \
           (cursor_pos[0] > x1 and cursor_pos[0] < x2) and \
           (cursor_pos[1] > y1 and cursor_pos[1] < y2):
            is_hovering = True
            currently_hovering_index = i

        text_color_pil = color_normal_pil
        bg_color_pil = None
        if i == dragged_object_index:
            text_color_pil = color_drag_pil
        elif is_hovering and right_hand_gesture == "POINT":
            text_color_pil = color_hover_pil

        if use_pillow_font:
            draw_y_pil = display_y_baseline - text_h
            if bg_color_pil:
                padding = 2
                draw.rectangle([(x1 - padding, y1 - padding), (x2 + padding, y2 + padding)], fill=bg_color_pil)

            draw.text((display_x, draw_y_pil), letter, font=font_sentence_pil, fill=text_color_pil)
        else:
            color_cv = (255, 255, 255)
            if i == dragged_object_index: color_cv = (0, 255, 0)
            elif is_hovering: color_cv = (0, 255, 255)
            cv2.putText(frame, letter, (display_x, display_y_baseline), font_face_cv, font_scale_sent_cv, color_cv, font_thickness_sent_cv)
    can_delete_now = time.time() > delete_cooldown_start

    if currently_hovering_index != -1:
        if right_hand_gesture == "POINT":
            if hover_target_index != currently_hovering_index:
                hover_target_index = currently_hovering_index
                hover_start_time = time.time()
            else:
                if hover_start_time is not None:
                    elapsed_hover_time = time.time() - hover_start_time
                    if elapsed_hover_time > HOVER_DELETE_SEC and can_delete_now:
                        delete_index = hover_target_index
                        print(f"Delete condition met for index {delete_index}")
                        hover_start_time = None
                        hover_target_index = -1
                        delete_cooldown_start = time.time() + DELETE_COOLDOWN_SEC 
        else:
            if hover_target_index != -1 or hover_start_time is not None:
                 hover_start_time = None
                 hover_target_index = -1

    else:
        if hover_target_index != -1 or hover_start_time is not None:
             hover_start_time = None
             hover_target_index = -1

    if delete_index != -1:
         if delete_index < len(letter_objects):
             deleted_char = letter_objects.pop(delete_index)['char']
             print(f"Deleted letter '{deleted_char}' by hovering with POINT.")
             if dragged_object_index != -1:
                 if delete_index == dragged_object_index: dragged_object_index = -1
                 elif delete_index < dragged_object_index: dragged_object_index -= 1

    if use_pillow_font:
        frame_display = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        frame_display = frame

    cv2.circle(frame_display, cursor_pos, 8, (0, 0, 255), -1)
    cv2.circle(frame_display, grab_cursor_pos, 10, (255, 0, 0), 2)

    cv2.imshow('Interpreter - Custom Font & Floating', frame_display)

cap.release()
cv2.destroyAllWindows()
final_sentence_list = [obj['char'] for obj in letter_objects]
print(f"Final objects: {''.join(final_sentence_list)}")