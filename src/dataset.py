import torch
import os
import xml.etree.ElementTree as ET
from PIL import Image
import torchvision.transforms.v2 as T # Используем transforms v2 для лучшей работы с bbox
from torchvision import tv_tensors

class CustomObjectDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annotation_dir, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transforms = transforms
        self.image_files = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.class_names = ['BACKGROUND', 'pallet', 'dome', 'Human'] 
        self.class_map = {name: i for i, name in enumerate(self.class_names)}

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        annotation_name = os.path.splitext(img_name)[0] + '.xml'
        annotation_path = os.path.join(self.annotation_dir, annotation_name)

        img = Image.open(img_path).convert("RGB")
        original_width, original_height = img.size

        boxes_list = []
        labels_list = []
        
        if os.path.exists(annotation_path):
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            for member in root.findall('object'):
                class_name = member.find('name').text

                xmin = int(member.find('bndbox').find('xmin').text)
                ymin = int(member.find('bndbox').find('ymin').text)
                xmax = int(member.find('bndbox').find('xmax').text)
                ymax = int(member.find('bndbox').find('ymax').text)
                
                if xmax <= xmin or ymax <= ymin:
                    print(f"Предупреждение: Невалидная рамка для класса '{class_name}' в файле {annotation_name}: ({xmin, ymin, xmax, ymax}). Пропускается.")
                    continue
                
                # Ограничиваем рамки границами изображения, если они выходят за пределы
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(original_width, xmax)
                ymax = min(original_height, ymax)

                if xmax <= xmin or ymax <= ymin: #повторная проверка после клиппинга
                    print(f"Предупреждение: Рамка стала невалидной после клиппинга для класса '{class_name}' в файле {annotation_name}. Пропускается.")
                    continue

                boxes_list.append([xmin, ymin, xmax, ymax])
                labels_list.append(self.class_map[class_name])
        
        if len(boxes_list) > 0:
            boxes_tensor = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels_list, dtype=torch.int64)
            target_boxes = tv_tensors.BoundingBoxes(
                boxes_tensor, 
                format=tv_tensors.BoundingBoxFormat.XYXY, 
                canvas_size=(original_height, original_width)
            )
        else:
            target_boxes = tv_tensors.BoundingBoxes(
                torch.zeros((0, 4), dtype=torch.float32),
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=(original_height, original_width)
            )
            labels_tensor = torch.zeros(0, dtype=torch.int64)

        target = {}
        target["boxes"] = target_boxes
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])
        
        #для оригинальных рамок
        area_data = target["boxes"].data 
        if area_data.numel() > 0:
            target["area"] = (area_data[:, 2] - area_data[:, 0]) * (area_data[:, 3] - area_data[:, 1])
        else:
            target["area"] = torch.zeros(0, dtype=torch.float32)
        
        target["iscrowd"] = torch.zeros((target["boxes"].shape[0],), dtype=torch.int64)

        if self.transforms:
            img, target = self.transforms(img, target)
        
        #после трансформаций target["boxes"] все еще объект BoundingBoxes,но с обновленными координатами и canvas_size

        return img, target

    def __len__(self):
        return len(self.image_files)

if __name__ == '__main__':
    IMG_FOLDER_PATH = r"C:\Users\lenovo\Malekwhat\ML.-KAGGLE\VideoSegmentation\data\processed_data\images"
    ANN_FOLDER_PATH = r"C:\Users\lenovo\Malekwhat\ML.-KAGGLE\VideoSegmentation\data\processed_data\annotations"

    print(f"Используем изображения из: {IMG_FOLDER_PATH}")
    print(f"Используем аннотации из: {ANN_FOLDER_PATH}")


    test_transforms = T.Compose([
        T.Resize((320, 320), antialias=True),
        T.PILToTensor(),                         
        T.ConvertImageDtype(torch.float32)         
    ])

    dataset = CustomObjectDetectionDataset(image_dir=IMG_FOLDER_PATH, 
                                         annotation_dir=ANN_FOLDER_PATH, 
                                         transforms=test_transforms)
    
    if len(dataset) == 0:
        print("Датасет пуст")
    else:
        print(f"Успешно найдено {len(dataset)} изображений в датасете.")
        print("\nПопытка загрузки первого элемента датасета (dataset[0])...")
        try:
            img, target = dataset[0]
            print("Успешно загружен первый элемент.")
            print(f"Изображение (форма тензора): {img.shape}, тип: {img.dtype}")
            print(f"Цель (target):")
            for k, v in target.items():
                if isinstance(v, tv_tensors.BoundingBoxes):
                    print(f"{k}: {v} (canvas_size: {v.canvas_size}, format: {v.format})")
                elif isinstance(v, torch.Tensor):
                    print(f"{k}: shape {v.shape}, dtype {v.dtype}")
                else:
                    print(f"{k}: {v}")
            
            if len(target['labels']) > 0:
                unique_labels = torch.unique(target['labels'])
                print(f"Уникальные числовые метки классов в первом элементе: {unique_labels.tolist()}")
                for label_idx in unique_labels:
                    print(f"Метка {label_idx.item()} соответствует классу: {dataset.class_names[label_idx.item()]}")
            else:
                print("В первом элементе нет размеченных объектов.")
            
        except Exception as e:
            print(f"ОШИБКА при загрузке или обработке первого элемента: {e}")
            import traceback
            traceback.print_exc() 