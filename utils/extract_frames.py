import cv2
import os

def extract_frames_from_mov(videos_folder):

    for file_name in os.listdir(videos_folder):
        if file_name.endswith('.MOV'):

            video_path = os.path.join(videos_folder, file_name)
            
            video_name = os.path.splitext(file_name)[0]
            output_folder = os.path.join(videos_folder, video_name)
            print(f"Output folder: {output_folder}")

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            video_capture = cv2.VideoCapture(video_path)
            success, frame = video_capture.read()
            frame_count = 0

            while success:

                frame_file_name = f"frame_{frame_count:04d}.jpg"
                frame_path = os.path.join(output_folder, frame_file_name)
                cv2.imwrite(frame_path, frame)

                success, frame = video_capture.read()
                frame_count += 1

            video_capture.release()

if __name__ == "__main__":

    videos_folder = r"C:\Users\works\OneDrive\Desktop\projects\vision\images\cvc"
    
    extract_frames_from_mov(videos_folder)
