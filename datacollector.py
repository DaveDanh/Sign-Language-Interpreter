import cv2
import mediapipe as mp
import numpy as np
import csv
import os

def load_counts(file_path):
    counts = {}
    if os.path.isfile(file_path):
        with open(file_path, 'r', newline='') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                label_index = header.index('label')
            except (StopIteration, ValueError):
                return counts 

            for row in reader:
                if len(row) > label_index:
                    label = row[label_index]
                    counts[label] = counts.get(label, 0) + 1
    return counts

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)

CSV_FILE_PATH = 'data.csv' 
file_exists = os.path.isfile(CSV_FILE_PATH)

# Tải bộ đếm
label_counts = load_counts(CSV_FILE_PATH)
print("Số lượng mẫu đã thu thập:", label_counts)

csv_file = open(CSV_FILE_PATH, 'a', newline='')
writer = csv.writer(csv_file)

if not file_exists or os.path.getsize(CSV_FILE_PATH) == 0:
    header = []
    for i in range(21):
        header += [f'x{i}', f'y{i}', f'z{i}']
    header.append('label')
    writer.writerow(header)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    image_rgb.flags.writeable = False
    results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    current_label = None
    key = cv2.waitKey(5) & 0xFF

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        data_row_relative = []
        wrist_landmark = hand_landmarks.landmark[0]
        wrist_x, wrist_y, wrist_z = wrist_landmark.x, wrist_landmark.y, wrist_landmark.z

        for lm in hand_landmarks.landmark:
            data_row_relative.extend([lm.x - wrist_x, lm.y - wrist_y, lm.z - wrist_z])
        
        max_abs_value = np.max(np.abs(data_row_relative))
        
        if max_abs_value == 0:
            continue
            
        data_row_normalized = (np.array(data_row_relative) / max_abs_value).tolist()

        if key != 255:
            if ord('a') <= key <= ord('z'):
                current_label = chr(key).upper()
            elif key == ord('1'):
                current_label = "SPACE"
            elif key == ord('2'):
                current_label = "STOP"
            elif key == ord('3'):
                current_label = "RESUME"
            elif key == ord('4'):
                current_label = "POINT"
            elif key == ord('0'):
                break
            
            if current_label is not None:
                writer.writerow(data_row_normalized + [current_label])
                label_counts[current_label] = label_counts.get(current_label, 0) + 1
                print(f"Đã lưu: {current_label} (Tổng: {label_counts[current_label]})")

    elif key == ord('0'):
        break

    cv2.imshow('Cong cu Thu Thap Du Lieu (Chuan Hoa Ty Le)', frame)

cap.release()
csv_file.close()
cv2.destroyAllWindows()