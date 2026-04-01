from utils import read_video, save_video

def main():
    #Read video
    video_frames = read_video('input_videos/TestVideo1.mp4')

    #Save video
    save_video(video_frames,'output_videos/run.avi')

if __name__ == '__main__':
    main()