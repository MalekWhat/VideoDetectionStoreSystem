import torch
import torchvision.transforms.v2 as T
import cv2
import numpy as np
import argparse
import os
import time
from PIL import Image

from model import get_model 

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
IMAGE_SIZE = 320


CLASSES = ['BACKGROUND', 'pallet', 'dome', 'Human'] 
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
COLORS[0] = [0, 0, 0] #чёрный фон

def get_transforms_for_detection():
    """
    Возвращает трансформации, которые нужно применить к кадру перед подачей в модель.
    Они теперь ожидают PIL Image на вход, чтобы соответствовать процессу обучения.
    """
    transforms = []
    transforms.append(T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True))
    transforms.append(T.PILToTensor()) 
    transforms.append(T.ConvertImageDtype(torch.float32))
    return T.Compose(transforms)

def preprocess_frame(frame_np):
    """
    Предобработка одного кадра: OpenCV BGR (H, W, C) numpy -> PyTorch CHW tensor.
    Конвертирует в PIL Image перед трансформацией.
    """
    rgb_frame_np = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB) # HWC, uint8
    pil_image = Image.fromarray(rgb_frame_np)
    transform = get_transforms_for_detection()
    transformed_frame = transform(pil_image)
    return transformed_frame

def predict(model, image_tensor, detection_threshold):
    """
    Делает предсказание на одном изображении.
    """
    image_tensor = image_tensor.to(DEVICE)
    
    with torch.no_grad():
        outputs = model([image_tensor])
    
    # outputs[0] содержит словарь с 'boxes', 'labels', 'scores'
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    pred_boxes = outputs[0]['boxes'].detach().cpu().numpy()
    pred_labels_indices = outputs[0]['labels'].detach().cpu().numpy()
    

    high_confidence_indices = np.where(pred_scores >= detection_threshold)[0]

    filtered_boxes = pred_boxes[high_confidence_indices]
    filtered_labels_indices = pred_labels_indices[high_confidence_indices]
    filtered_scores = pred_scores[high_confidence_indices]
    
    return filtered_boxes, filtered_labels_indices, filtered_scores

def draw_boxes_and_log(frame, boxes, labels_indices, scores, original_h, original_w, log_events=True):
    """
    Отрисовывает рамки на кадре и логирует события.
    """
    detected_objects_current_frame = set()

    for i, box in enumerate(boxes):
        label_idx = labels_indices[i]
        class_name = CLASSES[label_idx]
        score = scores[i]
        
        xmin, ymin, xmax, ymax = box

        x_scale = original_w / IMAGE_SIZE
        y_scale = original_h / IMAGE_SIZE
        
        xmin_orig = int(round(xmin * x_scale)) 
        ymin_orig = int(round(ymin * y_scale)) 
        xmax_orig = int(round(xmax * x_scale))
        ymax_orig = int(round(ymax * y_scale))
        
        xmin_orig = max(0, xmin_orig)
        ymin_orig = max(0, ymin_orig)
        xmax_orig = min(original_w -1, xmax_orig)
        ymax_orig = min(original_h -1, ymax_orig)

        color = COLORS[label_idx]
        cv2.rectangle(frame, (xmin_orig, ymin_orig), (xmax_orig, ymax_orig), color, 2)
        
        text = f"{class_name}: {score:.2f}"
        cv2.putText(frame, text, (xmin_orig, ymin_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if log_events:
            event_type = f"{class_name} detected"
            detected_objects_current_frame.add(class_name)
            print(f"LOG: {time.strftime('%Y-%m-%d %H:%M:%S')} - Event: {event_type}, Score: {score:.2f}, Box: [{xmin_orig}, {ymin_orig}, {xmax_orig}, {ymax_orig}]")

    return frame

def main(args):

    num_model_classes = len(CLASSES) 
    model = get_model(num_classes=num_model_classes, pretrained=False)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))

    model.to(DEVICE)
    model.eval()

    cap = cv2.VideoCapture(args.video_path)

    original_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Параметры видео: {original_w}x{original_h} @ {fps:.2f} FPS")

    #видео-рекордер для сохранения результата
    if args.output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # или 'XVID'
        out_video = cv2.VideoWriter(args.output_video_path, fourcc, fps, (original_w, original_h))
        print(f"Сохранено в: {args.output_video_path}")

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось или не удалось прочитать кадр.")
            break

        frame_count += 1
        input_tensor = preprocess_frame(frame.copy()) # frame.copy() чтобы не изменять оригинал

        boxes, labels_indices, scores = predict(model, input_tensor, args.threshold)
        processed_frame = draw_boxes_and_log(frame, boxes, labels_indices, scores, original_h, original_w)
        
        cv2.imshow('Real-time Object Detection', processed_frame)

        if args.output_video_path:
            out_video.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Выход по нажатию 'q'.")
            break
            
    end_time = time.time()
    processing_fps = frame_count / (end_time - start_time) if (end_time - start_time) > 0 else 0
    print(f"Обработка завершена. Средний FPS: {processing_fps:.2f}")

    cap.release()
    if args.output_video_path and out_video:
        out_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Детекция объектов на видео с использованием обученной модели SSDLite-MobileNetV3")
    parser.add_argument('--model_path', type=str, required=True, help='Путь к файлу обученной модели (.pth)')
    parser.add_argument('--video_path', type=str, required=True, help='Путь к видеофайлу для детекции')
    parser.add_argument('--output_video_path', type=str, default=None, help='(Опционально) Путь для сохранения обработанного видео')
    parser.add_argument('--threshold', type=float, default=0.5, help='Порог уверенности для отображения детекций (0.0 до 1.0)')
    
    args = parser.parse_args([
        '--model_path', r"C:\Users\lenovo\Malekwhat\ML.-KAGGLE\VideoSegmentation\models\ssdlite_mobilenetv3_custom_final.pth",
        '--video_path', r"C:\Users\lenovo\Malekwhat\ML.-KAGGLE\VideoSegmentation\data\video\video1.mp4",
        # '--output_video_path', 'output/detected_video2.mp4', # Раскомментируйте и укажите путь, если хотите сохранять
        '--threshold', "0.4"
    ])
    

    main(args) 