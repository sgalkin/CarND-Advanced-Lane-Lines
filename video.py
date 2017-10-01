#!/usr/bin/env python

from pipeline import Pipeline
import moviepy.editor as mpy
import argparse

def recode(video, camera):
    p = Pipeline(camera, video.size[::-1])
    return video.fl_image(lambda i: p.apply(i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video converseion utility')
    parser.add_argument('video', type=str,
                        help='Input video file')
    parser.add_argument('-o', dest='filename', type=str, required=True,
                        help='Output video file')
    parser.add_argument('-c', dest='camera', type=str, required=True,
                        help='Camera calibration parameters')
    parser.add_argument('-b', type=int, default=0, help='Start time')
    parser.add_argument('-e', type=int, default=-1, help='End time')
    args = parser.parse_args()

    video = mpy.VideoFileClip(args.video)
    video = video.subclip(args.b, args.e)
    video = recode(video, args.camera)
    video.write_videofile(args.filename, audio=False)
    
