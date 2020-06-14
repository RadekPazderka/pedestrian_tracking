import os

import cv2
import time
import argparse
import wget
from detector import Detector
from tracker import Tracker

def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", help="input video path", required=True, type=str)
    parser.add_argument("--output_video_path", help="output video path (if not filled so output will be printed to screen)",
                        default="", type=str)
    args = parser.parse_args()
    return args


def process_video(video_path, output_video=None):
    if not os.path.exists("yolov3.weights"):
        print("downloading checkpoint yolov3.weights... wait please.")
        wget.download("https://pjreddie.com/media/files/yolov3.weights")
    det_inst = Detector("yolov3.weights", "yolov3.cfg")

    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    tracker_inst = Tracker()
    frames = 0

    if output_video is not None:
        out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'.mp4'), float(15), (int(width), int(height)))

    while True:
        frame = cap.read()[1]
        if frame is None:
           break
        #frame= cv2.resize(frame, (414,414))

        while True:
            if frames % 5 == 0:
                res, bboxes, _ = det_inst.detect_image(frame)
                tracker_inst.update_trackers_by_dets(frame, bboxes)
            frame = cap.read()[1]

            if frame is None:
                break

            start = time.time()
            annoted_frame = tracker_inst.track(frame)
            if output_video is not None:
                out.write(annoted_frame)
            else:
                #show the output frame
                cv2.imshow("Output", frame)
                cv2.waitKey(1)

            frames += 1
            print("Frame: {0}, time: {1}".format(frames, time.time() - start))
    if output_video:
        out.release()


if __name__ == '__main__':
    args = parse_parameters()
    process_video(args.video_path, args.output_video_path)