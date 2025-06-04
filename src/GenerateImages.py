import cv2
import os

def extract_frames(video_path, output_folder, frame_skip):
    """
    Извлекает кадры из видео и сохраняет их в указанную папку.
    video_path (str): Путь к видеофайлу.
    output_folder (str): Папка для сохранения извлеченных кадров.
    frame_skip (int): Сохранять каждый N-ный кадр.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break 

        if frame_count % frame_skip == 0:

            frame_filename = os.path.join(output_folder, f"frame_{saved_frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1
            print(f"Сохранен кадр: {frame_filename}")

        frame_count += 1

    cap.release()
    print(f"Извлечение завершено. Всего сохранено {saved_frame_count} кадров из {video_path}")

video_file_1 = r"C:\Users\lenovo\Malekwhat\ML.-KAGGLE\VideoSegmentation\data\video\video1.mp4"
output_dir_1 = r"C:\Users\lenovo\Malekwhat\ML.-KAGGLE\VideoSegmentation\data\images\ImFromV1"
frames_to_skip = 30 #1 кадр в секунду для 30fps видео

extract_frames(video_file_1, output_dir_1, frames_to_skip)