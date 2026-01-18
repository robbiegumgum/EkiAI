import sys
import time
import threading
import queue
import math
import os
from collections import defaultdict, deque
import logging

CAMERA_INDEX = 3
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
AMBULANCE_TRIGGER_TIME = 4.0
STAFF_LOOP_INTERVAL = 6.0
HIP_INFLUENCE = 0.80
SHOULDER_INFLUENCE = 1.0 - HIP_INFLUENCE
NOISE_FLOOR = 2.0
MOTION_THRESHOLD = 3.0

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import pyttsx3
    import pygame
    from PIL import Image
    from transformers import pipeline
    logging.getLogger("transformers").setLevel(logging.ERROR)
except ImportError as e:
    sys.exit(1)

current_mode = 0
mode_names = ["STAFF ALERT", "AMBULANCE"]
latest_danger_status = "SAFE"
latest_motion_condition = "ANALYZING..."
latest_visual_description = "Scanning..."
trigger_ai_analysis = False
frame_for_ai_analysis = None
is_ambulance_called = False
active_track_id = None
track_histories = {}
track_positions = {}
transcript_history = ["System initialized."]

def update_transcript(text):
    global transcript_history
    timestamp = time.strftime("%H:%M:%S")
    lines = text.split('\n')
    for line in lines:
        if line.strip():
            transcript_history.append(f"{timestamp} {line}")
    if len(transcript_history) > 10:
        transcript_history = transcript_history[-10:]

audio_queue = queue.Queue()

def audio_worker():
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
    except:
        return
    while True:
        try:
            text = audio_queue.get()
            if text is None: break
            engine.say(text)
            engine.runAndWait()
            audio_queue.task_done()
        except:
            audio_queue.task_done()

def speak_text(text):
    audio_queue.put(text)

def wait_until_speech_finished():
    audio_queue.join()

def play_mode_sound(mode_index):
    def worker():
        try:
            file_map = {0: "ring.mp3", 1: "ambulance.mp3"}
            filename = file_map.get(mode_index)
            
            if filename and os.path.exists(filename):
                pygame.mixer.music.load(filename)
                pygame.mixer.music.set_volume(0.5)
                pygame.mixer.music.play()
                time.sleep(2.0)
                pygame.mixer.music.stop()
        except Exception:
            pass
    threading.Thread(target=worker, daemon=True).start()

captioner = None

def setup_system():
    global captioner
    pygame.mixer.init()
    try:
        captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    except Exception:
        pass

def visual_description_worker():
    global latest_visual_description, frame_for_ai_analysis, trigger_ai_analysis, captioner
    while True:
        if trigger_ai_analysis and frame_for_ai_analysis is not None:
            if captioner is None:
                trigger_ai_analysis = False
                time.sleep(1)
                continue
            try:
                rgb_frame = cv2.cvtColor(frame_for_ai_analysis, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                result = captioner(pil_img, max_new_tokens=25)[0]['generated_text']
                clean_desc = result.replace("a photograph of ", "").strip()
                latest_visual_description = clean_desc
                trigger_ai_analysis = False
            except Exception:
                trigger_ai_analysis = False
                time.sleep(1.0)
        else:
            time.sleep(0.1)

def calculate_motion_state(track_id, curr_x, curr_y):
    global track_histories, track_positions
    if track_id not in track_histories:
        track_histories[track_id] = deque(maxlen=30)
        track_positions[track_id] = (curr_x, curr_y)
        return "analyzing"

    prev_x, prev_y = track_positions[track_id]
    dist = math.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
    track_positions[track_id] = (curr_x, curr_y)

    if dist < NOISE_FLOOR: dist = 0.0
    track_histories[track_id].append(dist)

    if len(track_histories[track_id]) < 15:
        return "analyzing"

    avg_movement = sum(track_histories[track_id]) / len(track_histories[track_id])

    if avg_movement > MOTION_THRESHOLD:
        return "struggling"
    else:
        return "unconscious"

def trigger_ambulance_bundled(location, status):
    global is_ambulance_called, latest_visual_description
    if is_ambulance_called: return
    is_ambulance_called = True
    
    wait_start = time.time()
    while (time.time() - wait_start) < 2.0:
        if "Scanning" not in latest_visual_description:
            break
        time.sleep(0.1)

    update_transcript(f"Alert: {status.upper()}")

    desc_text = ""
    if "Scanning" not in latest_visual_description:
        update_transcript(f"Description: {latest_visual_description}")
        desc_text = f"Description: {latest_visual_description}."

    full_msg = f"Emergency at {location}. Status is {status}. {desc_text}"
    speak_text(full_msg)
    
    wait_until_speech_finished()
    speak_text("Dispatching unit immediately.")
    update_transcript("Dispatching unit")

def staff_alert_worker():
    global latest_danger_status, current_mode, latest_motion_condition, latest_visual_description
    while True:
        if current_mode == 0 and "CRITICAL" in latest_danger_status:
            status_clean = latest_motion_condition.lower()
            if "analyzing" in status_clean:
                time.sleep(0.1)
                continue

            msg_status = f"Emergency at Platform 1. Status: {status_clean}."
            msg_desc = ""
            if "Scanning" not in latest_visual_description:
                msg_desc = f"Description: {latest_visual_description}."

            update_transcript(f"Alert: {status_clean.upper()}")
            if msg_desc:
                update_transcript(msg_desc)

            full_audio = f"{msg_status} {msg_desc}"
            speak_text(full_audio)
            
            wait_until_speech_finished()
            time.sleep(STAFF_LOOP_INTERVAL)
        else:
            time.sleep(0.1)

def draw_ui(frame, line_x, status, color, height, width, track_id):
    cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 2)
    cv2.putText(frame, "PLATFORM", (line_x - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, "TRACKS", (line_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.rectangle(frame, (0, height-60), (width, height), color, -1)
    
    status_text = f"STATUS: {status}"
    text_col = (0, 0, 0) if "WARNING" in status else (255, 255, 255)
    cv2.putText(frame, status_text, (20, height-20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_col, 2)
    
    mode_color = (0, 255, 255) if current_mode == 0 else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (width, 50), (50, 50, 50), -1)
    cv2.putText(frame, f"MODE: {mode_names[current_mode]} Press ENTER", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
    
    if "SAFE" not in status and "WARNING" not in status:
        cv2.rectangle(frame, (width//2, 80), (width, 160), (0, 0, 0), -1)
        display_cond = latest_motion_condition.upper()
        cv2.putText(frame, f"COND: {display_cond}", (width//2 + 10, 105), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        desc_short = latest_visual_description[:25] + "..." if len(latest_visual_description) > 25 else latest_visual_description
        cv2.putText(frame, f"DESC: {desc_short}", (width//2 + 10, 135),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

def draw_log_window():
    log_img = np.zeros((450, 1100, 3), dtype=np.uint8)
    cv2.putText(log_img, "System Log", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.line(log_img, (20, 40), (1080, 40), (100, 100, 100), 1)
    
    y = 80
    for i, line in enumerate(transcript_history):
        col = (200, 200, 200)
        if i == len(transcript_history) - 1:
            col = (255, 255, 255)
        
        if "Description:" in line:
            col = (255, 200, 100)
        elif "Alert" in line:
            col = (100, 100, 255)
            
        cv2.putText(log_img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1)
        y += 35
    
    cv2.imshow('Log', log_img)

def main():
    global latest_danger_status, frame_for_ai_analysis, trigger_ai_analysis, current_mode, is_ambulance_called, latest_motion_condition, active_track_id
    
    threading.Thread(target=audio_worker, daemon=True).start()
    setup_system()
    threading.Thread(target=staff_alert_worker, daemon=True).start()
    threading.Thread(target=visual_description_worker, daemon=True).start()
    
    try:
        model = YOLO('yolov8n-pose.pt')
    except Exception:
        return

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    danger_start_time = None
    skeleton_links = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
                      (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        line_x = width // 2
        
        results = model.track(frame, persist=True, verbose=False)
        highest_priority_state = "SAFE"
        highest_priority_color = (0, 200, 0)
        active_track_id = None
        
        if results[0].boxes.id is not None and results[0].keypoints is not None:
            track_ids = results[0].boxes.id.int().cpu().numpy()
            keypoints_data = results[0].keypoints.data
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            for track_id, keypoints, box in zip(track_ids, keypoints_data, boxes):
                if keypoints.shape[0] == 0: continue
                kps = keypoints.cpu().numpy()
                
                if kps[5][2] > 0.5 and kps[11][2] > 0.5:
                    shoulder_x_avg = (kps[5][0] + kps[6][0]) / 2
                    hip_x_avg = (kps[11][0] + kps[12][0]) / 2
                    cog_x = int((hip_x_avg * HIP_INFLUENCE) + (shoulder_x_avg * SHOULDER_INFLUENCE))
                    cog_y = int((kps[11][1] + kps[12][1]) / 2)
                    
                    cv2.circle(frame, (cog_x, cog_y), 6, (255, 0, 255), -1)
                    
                    any_part_over_line = False
                    for kp in kps:
                        if kp[2] > 0.5 and kp[0] > line_x:
                            any_part_over_line = True
                            break

                    cog_in_danger = cog_x > line_x
                    person_state = "SAFE"
                    person_color = (0, 200, 0)

                    if cog_in_danger:
                        person_state = "CRITICAL"
                        person_color = (0, 0, 255)
                    elif any_part_over_line:
                        person_state = "WARNING"
                        person_color = (0, 255, 255) 
                    
                    if person_state == "CRITICAL":
                        highest_priority_state = "CRITICAL: ON TRACKS"
                        highest_priority_color = (0, 0, 255)
                        active_track_id = track_id
                        
                        temp_motion = calculate_motion_state(track_id, cog_x, cog_y)
                        if temp_motion != "analyzing":
                            latest_motion_condition = temp_motion
                        
                        if not trigger_ai_analysis and not is_ambulance_called:
                            x1, y1, x2, y2 = map(int, box)
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(width, x2); y2 = min(height, y2)
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                frame_for_ai_analysis = crop.copy()
                                trigger_ai_analysis = True

                        if danger_start_time is None: danger_start_time = time.time()
                        elapsed = time.time() - danger_start_time
                        
                        if current_mode == 1: 
                            remaining = AMBULANCE_TRIGGER_TIME - elapsed
                            if remaining > 0:
                                cv2.putText(frame, f"CALL IN: {remaining:.1f}s", (cog_x, cog_y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            else:
                                if not is_ambulance_called:
                                    cv2.putText(frame, "CALLING...", (cog_x, cog_y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                                    final_status = latest_motion_condition
                                    if "analyzing" in final_status.lower(): final_status = "unconscious"
                                    threading.Thread(target=lambda: trigger_ambulance_bundled("Platform 1", final_status)).start()
                                else:
                                    cv2.putText(frame, "DISPATCHED", (cog_x, cog_y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    elif person_state == "WARNING":
                        if "CRITICAL" not in highest_priority_state:
                            highest_priority_state = "WARNING"
                            highest_priority_color = (0, 255, 255)
                            active_track_id = track_id
                            danger_start_time = None 
                            is_ambulance_called = False

                    for p1, p2 in skeleton_links:
                        if kps[p1][2] > 0.5 and kps[p2][2] > 0.5:
                            pt1 = (int(kps[p1][0]), int(kps[p1][1]))
                            pt2 = (int(kps[p2][0]), int(kps[p2][1]))
                            cv2.line(frame, pt1, pt2, person_color, 2)
                    
                    cv2.putText(frame, f"ID:{track_id}", (cog_x, cog_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if highest_priority_state == "SAFE":
            danger_start_time = None
            is_ambulance_called = False 
        
        latest_danger_status = highest_priority_state
        
        draw_ui(frame, line_x, highest_priority_state, highest_priority_color, height, width, active_track_id)
        draw_log_window()
        cv2.imshow('Camera Feed', frame)
        
        if frame_for_ai_analysis is not None:
            cv2.imshow('AI Input', frame_for_ai_analysis)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'): break
        elif key == 13: 
            current_mode = 1 - current_mode
            is_ambulance_called = False
            play_mode_sound(current_mode)
            speak_text(f"Mode switched to {mode_names[current_mode]}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()