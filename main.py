from utils import read_video, save_video
from trackers import Tracker
from ultralytics import YOLO

def main():

    input_video = 'input_videos/TestVideo1.mp4'
    output_video = 'output_videos'

    model = YOLO('models/best.pt')
    #Read video
    video_frames = read_video(input_video)
    
    tracker = Tracker('models/best.pt')
    tracks = tracker.get_oject_tracks(video_frames)

    #Save video
    save_video(video_frames,output_dir=output_video)

if __name__ == '__main__':
    main()