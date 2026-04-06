from utils import read_video, save_video
from trackers import Tracker
from ultralytics import YOLO
import cv2
def main():

    input_video = 'input_videos/TestVideo1.mp4'
    output_video = 'output_videos'
    model = 'models/best_yolo11s.pt'

    #Read video
    video_frames = read_video(input_video)
    
    # Init Tracker
    tracker = Tracker(model)
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    for track_id, player in tracks['players'][3].items():
        bbox = player['bbox']
        frame = video_frames[3]

        cropped = frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]

        cv2.imwrite(f'output_videos/cropped_img3.jpg',cropped)
        break
    #Draw ouput objects tracked
    #output_video_frames = tracker.draw_annotations(video_frames,tracks)

    #Save video
    #save_video(output_video_frames,output_dir=output_video)

if __name__ == '__main__':
    main()