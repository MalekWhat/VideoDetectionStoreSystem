import torch
import os
import time
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

from dataset import CustomObjectDetectionDataset
from model import get_model


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # VideoSegmentation/
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed_data')
IMG_DIR = os.path.join(DATA_DIR, 'images')
ANN_DIR = os.path.join(DATA_DIR, 'annotations')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_CLASSES = 4 # включая фон (BACKGROUND, pallet, dome, human)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Используемое устройство: {DEVICE}")

BATCH_SIZE = 2
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
SAVE_MODEL_PATH = os.path.join(OUTPUT_DIR, 'ssdlite_mobilenetv3_custom_final.pth')

TRAIN_SPLIT_RATIO = 0.8
IMAGE_SIZE = 320


def get_transforms(train=False):
    transforms = []
    transforms.append(T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True))
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float32))

    return T.Compose(transforms)

def collate_fn(batch):
    """
    Функция для правильной сборки батчей для моделей детекции объектов torchvision.
    Она просто возвращает кортеж из списка изображений и списка целей.
    """
    return tuple(zip(*batch))

def main():

    train_transforms = get_transforms(train=True)
    val_transforms = get_transforms(train=False)

    dataset_for_train = CustomObjectDetectionDataset(image_dir=IMG_DIR, annotation_dir=ANN_DIR, transforms=train_transforms)
    dataset_for_val = CustomObjectDetectionDataset(image_dir=IMG_DIR, annotation_dir=ANN_DIR, transforms=val_transforms)
    
    if len(dataset_for_train) == 0:
        print("Датасет пуст")
        return

    num_total_samples = len(dataset_for_train)
    num_train_samples = int(TRAIN_SPLIT_RATIO * num_total_samples)
    num_val_samples = num_total_samples - num_train_samples

    print(f"Всего изображений: {num_total_samples}")
    print(f"Обучающая выборка: {num_train_samples} изображений")
    print(f"Валидационная выборка: {num_val_samples} изображений")


    indices = torch.randperm(num_total_samples).tolist()
    
    train_subset = torch.utils.data.Subset(dataset_for_train, indices[:num_train_samples])
    val_subset = torch.utils.data.Subset(dataset_for_val, indices[num_train_samples:])

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
        drop_last=False
    )
 
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model.to(DEVICE)
    print("Модель инициализирована")

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    
    
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
  


    print("Начало обучения...")
    start_train_time = time.time()
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        epoch_start_time = time.time()
        print(f"---Эпоха {epoch + 1}/{NUM_EPOCHS} ---")

        model.train()
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            optimizer.zero_grad() 

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values()) 

            if not torch.isfinite(losses):
                print(f"Бесконечные/NaN потери на {i}-ом шаге")
                print(f"Компоненты потерь: {loss_dict}")
                continue 

            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            
            if (i + 1) % 10 == 0: 
                print(f"[Эпоха {epoch + 1}, Батч {i + 1}/{len(train_loader)}] Тренировочные потери: {losses.item():.4f}")
        
        if len(train_loader) > 0:
            avg_train_loss = running_loss / len(train_loader)
            print(f"Средние тренировочные потери за эпоху {epoch + 1}: {avg_train_loss:.4f}")
        else:
            avg_train_loss = float('inf')


        val_running_loss = 0.0
        
        with torch.no_grad(): #отключаем вычисление градиентов
            for images_val, targets_val in val_loader:
                images_val = list(image.to(DEVICE) for image in images_val)
                targets_val = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets_val]

                loss_dict_val = model(images_val, targets_val) 
                

                losses_val = sum(loss for loss in loss_dict_val.values())
                val_running_loss += losses_val.item()

        if len(val_loader) > 0:
            avg_val_loss = val_running_loss / len(val_loader)
            print(f"Средние валидационные потери за эпоху {epoch + 1}: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                print(f"Валидационные потери улучшились ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Сохраняем модель")
                best_val_loss = avg_val_loss
                try:
                    torch.save(model.state_dict(), SAVE_MODEL_PATH)
                    print(f"Модель сохранена в: {SAVE_MODEL_PATH}")
                except Exception as e:
                    print(f"Ошибка при сохранении модели: {e}")
        else:
            print("Валидационный загрузчик пуст, пропускаем оценку валидационных потерь.")

        
        epoch_time_taken = time.time() - epoch_start_time
        print(f"Время на эпоху {epoch + 1}: {epoch_time_taken:.2f} сек.")

    total_train_time = time.time() - start_train_time
    print(f"Обучение завершено. Общее время: {total_train_time / 60:.2f} мин.")

    
    try:
        torch.save(model.state_dict(), SAVE_MODEL_PATH)
        print(f"Модель сохранена в: {SAVE_MODEL_PATH}")
    except Exception as e:
        print(f"Ошибка при сохранении модели: {e}")

if __name__ == '__main__':
    main() 