from utils import read_video, save_video
from trackers import Tracker
from ultralytics import YOLO
import cv2
from team_assign import TeamAssign
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

    # Assign Player Teams
    team_assigner = TeamAssign()
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):
        for player_id,track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                track['bbox'],
                                                player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    #Draw ouput objects tracked
    output_video_frames = tracker.draw_annotations(video_frames,tracks)

    #Save video
    save_video(output_video_frames,output_dir=output_video)

if __name__ == '__main__':
    main()