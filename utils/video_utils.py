import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame =cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(output_video_frames,output_video_path):
     
    if len(output_video_frames) == 0:
        print("No frames!")
        return

    if output_video_frames[0] is None:
        print("Invalid frame!")
        return

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)
    out.release()