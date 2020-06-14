# Pedestrian tracking
Simple tracking all pedestrian in video. Video tutorial here: https://youtu.be/cpyr_Y1_EIk

## 1) Conda instalation:
 - conda create -y --name pedestrian_tracking python==3.7
 - conda install -c conda-forge -y --name pedestrian_tracking --file requirements.txt

## 2) Download YOLOv3 weights to root project directory:
- wget https://pjreddie.com/media/files/yolov3.weights

## 3) Demo command:
- python3 demo.py

## 4) Run example:
- save output to video file: 
python3 main.py --video_path videos/soccer_01.mp4 --output_video_path "out.mp4"

- process video without result saving, only view:
python3 main.py --video_path videos/soccer_01.mp4


