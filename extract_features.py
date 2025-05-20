!git clone https://github.com/IIT-PAVIS/SPIGA.git
import sys
sys.path.append('./SPIGA')

import os
import cv2
import csv
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import mediapipe as mp

# === Config ===
VIDEO_PATH = '../data/input_video.mp4'
FRAME_DIR = '../data/frames/'
FACE_DIR = '../data/faces/'
CSV_PATH = '../data/features.csv'

# === Create folders ===
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(FACE_DIR, exist_ok=True)

# === Load models ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, device=device)  # only best face
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# === Open video ===
cap = cv2.VideoCapture(VIDEO_PATH)
frame_idx = 0
features_list = []

print("[INFO] Processing video...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Save frame
    frame_path = os.path.join(FRAME_DIR, f'frame_{frame_idx:04d}.jpg')
    cv2.imwrite(frame_path, frame)

    # Convert to PIL for MTCNN
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    box = mtcnn.detect(pil_img)[0]

    if box is not None:
        box = box[0]  # best face only
        x0, y0, x1, y1 = map(int, box)
        face = frame[y0:y1, x0:x1]
        face_path = os.path.join(FACE_DIR, f'face_{frame_idx:04d}.jpg')
        cv2.imwrite(face_path, face)

    # Pose estimation
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Extract keypoints
    keypoints = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        keypoints = [0.0] * (33 * 4)  # 33 pose landmarks * 4 values

    features_list.append([frame_idx] + keypoints)
    frame_idx += 1

cap.release()

# === Save to CSV ===
print("[INFO] Saving features to CSV...")
with open(CSV_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    header = ['frame_idx'] + [f'pose_{i}_{coord}' for i in range(33) for coord in ['x', 'y', 'z', 'v']]
    writer.writerow(header)
    writer.writerows(features_list)

print("[DONE] Feature extraction complete.")
