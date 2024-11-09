import argparse
from avi_to_mp4 import avi_2_mp4
import os, sys
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('path',type=str) # extri_calib folder
parser.add_argument('--pattern',type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])))
parser.add_argument('--grid', type=float)
args = parser.parse_args()

# 如果外參影片並非avi檔，則轉mp4檔
if os.path.isdir(args.path):
    extri_videos_path = join(args.path, 'videos')
    for filename in os.listdir(extri_videos_path):
        if filename.endswith(".avi"):
            avi_2_mp4(extri_videos_path)
            break

cmd_1 = f'python /MOCAP/EasyMocap/scripts/preprocess/extract_video.py {args.path} --no2d'
os.system(cmd_1)
cmd_2 = f'python /MOCAP/EasyMocap/scripts/preprocess/random_process.py {args.path}'
os.system(cmd_2)
chessboard = join(args.path, 'chessboard')
cmd_3 = f'python /MOCAP/EasyMocap/apps/calibration/detect_chessboard.py {args.path} --out {chessboard} --pattern {args.pattern[0]},{args.pattern[1]} --grid {args.grid}'
os.system(cmd_3)








