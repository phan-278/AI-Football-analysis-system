from utils import read_video, save_video
from trackers import Tracker
from ultralytics import YOLO
import cv2
from team_assign import TeamAssign, PlayerBallAssign
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_tranformer import ViewTransformer
from speed_estimator import SpeedAndDistance_Estimator
from reid import ReID
from tatical_map import TaticalMap
def main():

    input_video = 'input_videos/TestVideo1.mp4'
    output_video = 'output_videos'
    model = 'models/best_yolo11s.pt'
    model_pitch = 'models/best1_pitch.pt'
    reid_stub_path = 'stubs/reid_tracks_stubs.pkl'
    
    # == 1 ===================== READ VIDEO =====================
    video_frames = read_video(input_video)
    
    # == 2 ===================== TRACKING =====================
    # Get object track
    tracker = Tracker(model)
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # ADD object position
    tracker.add_position_to_tracks(tracks)


    # == 3 ===================== CAMERA MOVEMENT =====================
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub = True,
                                                                              stub_path = 'stubs/camera_movement.pkl')

    # Adjust object position
    camera_movement_estimator.add_adjust_position_to_tracks(tracks,camera_movement_per_frame)


    # == 4 ===================== VIEW TRANSFORM ====================
    #Add real position
    view_transformer = ViewTransformer(model_pitch_path=model_pitch)
    view_transformer.add_transformed_position_to_tracks(tracks, video_frames=video_frames)
    # Interpolate Ball Positions
        #BALL INTERPOLATION
    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    
    # == 5 ===================== TEAM ASSIGN =====================
    # Assign Player Teams
    team_assigner = TeamAssign()
    team_assigner.assign_team_color(video_frames[0],tracks['players'][0])
    
    for frame_num, player_track in enumerate(tracks['players']):

        for player_id,track in player_track.items():
            team = team_assigner.assign_player_into_team(video_frames[frame_num],
                                                         track['bbox'],
                                                         player_id)

            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    # == 6 ==================== ReID ==============================
    """
    reid = ReID(
        model_name           = 'osnet_x1_0',
        similarity_threshold = 0.75,
        jacket_dist_thresh   = 60,    # pixel
        jacket_time_thresh   = 48,    # frame (~2s)
        device               = 'cpu',
    )
    tracks = reid.merge_tracks_offline(video_frames, tracks)

    """
    # == 7 ===================== SPEED & DISTANCE =================
    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)


    # == 8 ========================= Ball possession ==================
    # Assign Ball Acquistition
    player_assigner = PlayerBallAssign()
    team_ball_control= []

    for frame_num, player_track in enumerate(tracks['players']):
        if 1 in tracks['ball'][frame_num]:
            ball_bbox = tracks['ball'][frame_num][1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)
        else:
            assigned_player = -1
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] =True
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])
        else:
            # if not anyone touch the ball set team 2 has ball by default
            if len(team_ball_control) > 0:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(2)
    team_ball_control = np.array(team_ball_control)


    # == 9 ===================== DRAW OUTPUT =====================
    # Draw ouput
    ## Draw ouput objects tracked
    output_video_frames = tracker.draw_annotations(video_frames,tracks,team_ball_control)

    ## Draw camera movemet
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,camera_movement_per_frame)
    
    ## Draw speed & distance
    speed_and_distance_estimator.draw_speed_and_distance( output_video_frames, tracks)

    # == 10 ===================== TACTICAL MAP PIP ====================
    # Overlay 2D Tactical Map
    tatical_map_generator = TaticalMap()
    output_video_frames = tatical_map_generator.draw_tatical_map_pip(output_video_frames, tracks)

    #Save video
    save_video(output_video_frames,output_dir=output_video)
    
    
if __name__ == '__main__':
    main()