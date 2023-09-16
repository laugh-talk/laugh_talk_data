import cv2
import os
import glob 
import argparse
import mediapipe as mp
import numpy as np
import time
import math
import pandas as pd
import shutil

parser = argparse.ArgumentParser(description="Final crop method num2")

parser.add_argument('--out_dir', type=str, default=None, help='Path for output')
parser.add_argument('--videoFolder', type=str, default=None, help='Path for inputs')

args = parser.parse_args()

args.out_dir = os.path.join(args.videoFolder,"processed_crop")
args.videoFolder = os.path.join(args.videoFolder,"processed_talk")

os.makedirs(args.out_dir,exist_ok=True)


def process_ffmpeg_actions(raw_vid_path, save_folder, save_vid_name, time_period):

    def secs_to_timestr(secs):
        hrs = secs // (60 * 60)
        min = (secs - hrs * 3600) // 60
        sec = secs % 60
        end = (secs - int(secs)) * 100
        return "{:02d}:{:02d}:{:02d}.{:02d}".format(int(hrs), int(min), int(sec), int(end))

    # seg ['laugh', ['0:05:16', '0:05:17', 1, 0]]
    for seg in time_period:
        out_path = os.path.join(save_folder, save_vid_name)

        cnt=0
        while os.path.exists(out_path):
            save_vid_name.split(".")[0]
            out_path = os.path.join(save_folder, save_vid_name.split(".")[0]+ str(cnt) + ".mp4")
            cnt += 1
        start_sec, end_sec = seg
        if start_sec != 0:
            start_sec +=0.3
        else:
            start_sec += 0.2
        end_sec -= 0.2
        cmd = f"ffmpeg -i {raw_vid_path} -ss {secs_to_timestr(start_sec)} -to {secs_to_timestr(end_sec)} -loglevel error {out_path}"
        os.system(cmd)



mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cel_text = glob.glob(args.videoFolder + '/*.mp4')

idx_lst = [33, 123, 133, 362, 263, 9, 6, 8, 1, 61, 291,199,352]
crop_time = {}

out_dir = args.out_dir
pth_len = len(args.videoFolder) + 1
cnt = 0
for video in cel_text:

    video_file = video 
    video_name= video_file[pth_len:-4]
    cap = cv2.VideoCapture(video_file) 
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        continue
    length = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = length/fps

    crop_time[video_name]= {'time': []}
    time = crop_time[video_name]
    time[video_name] = video_name
    
    c_time = 0.0
    p_time = 0.0
    cont_true = True
    remove_yn = True
    while cap.isOpened():
        ret, image = cap.read()
        if ret == False:
            break
        p_change = True
        c_change = True
        
        image= cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image)
        image= cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []
        
        if results.multi_face_landmarks:
            for face_landmark in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmark.landmark):
                    if idx in idx_lst: 
                        if idx == 1:
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                            x_nose_center = lm.x * img_w
                            y_nose_center = lm.y * img_h
                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                        elif idx == 9:
                                x_up_center = lm.x * img_w
                                y_up_center = lm.y * img_h
                
                        elif idx == 123:
                            x_right_cheek = lm.x * img_w
                            y_right_cheek = lm.y * img_h
                       
                        elif idx == 199:
                            x_down_center = lm.x * img_w
                            y_down_center = lm.y * img_h      

                        elif idx == 352:
                            x_left_cheek = lm.x * img_w
                            y_left_cheek = lm.y * img_h    

                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        face_2d.append([x,y])
                        face_3d.append([x,y, lm.z])
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                      [0, focal_length, img_w / 2],
                                      [0, 0, 0]])

                dist_matrix = np.zeros((4,1), dtype=np.float64)
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # get the y rotation degree
                x = angles[0] * 360

                nose_3d_projection, jaconian =cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                center_len = int(math.sqrt(math.pow(x_up_center - x_nose_center, 2) + math.pow(y_up_center - y_nose_center, 2)))
                width_len_right = int(math.sqrt(math.pow(x_right_cheek - x_nose_center, 2) + math.pow(y_right_cheek - y_nose_center, 2)))
                width_len_left = int(math.sqrt(math.pow(x_left_cheek - x_nose_center, 2) + math.pow(y_left_cheek - y_nose_center, 2)))
                x_diff = int(abs(x_left_cheek-x_nose_center))
                x_r_diff = int(abs(x_right_cheek-x_nose_center))

                if width_len_left == 0:
                    width_len_left = 0.01
                if width_len_right == 0:
                    width_len_right = 0.01
                width_portion_r = int((width_len_right/width_len_left) * 100)
                width_portion = int((width_len_left/width_len_right) * 100)

                    # False-> True
                if results.multi_face_landmarks == None:
                    if c_change == False:
                        continue

                    else:
                        cont_true = False
                        p_change = c_change
                        c_change = False
                        p_time = c_time
                        c_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                        if c_time - p_time > 3.5:
                            time['time'].append((p_time,c_time))

                if y_nose_center < 20 or (img_w-x_left_cheek) < 10 or x_right_cheek < 10:
                    remove_yn =False

                if width_portion < 10 or width_portion_r < 10 or x>20 or x < -13:
                    if c_change == False:
                        continue

                    else:
                        cont_true = False
                        p_change = c_change
                        c_change = False
                        p_time = c_time
                        c_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                        if c_time - p_time > 3.5:
                            time['time'].append((p_time,c_time-0.1))

                else:
                    if c_change == True:
                        continue

                    else:
                        p_change = c_change
                        c_change = True
                        p_time = c_time
                        c_time = cap.get(cv2.CAP_PROP_POS_MSEC)/1000
                        
        if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        
    cap.release()

    if (c_change == True) and (duration-c_time > 3.5):
        time['time'].append((c_time,duration))
    if remove_yn == False:
        del(crop_time[video_name])


for key, val in crop_time.items():
    raw_vid_path = args.videoFolder + "/" + key +".mp4"
    save_vid_name = key + '.mp4'
    time_period = val['time']
    process_ffmpeg_actions(raw_vid_path, out_dir, save_vid_name, time_period)

shutil.rmtree(args.videoFolder)
