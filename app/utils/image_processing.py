import numpy as np
import cv2
from PIL import Image


def detect_and_crop(pil_image, yolo_model, target_class=0):
    cropped_images = []
    img_np = np.array(pil_image)
    results = yolo_model(img_np)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0].item())
            if cls == target_class:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_np.shape[1], x2), min(img_np.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    cropped_img = img_np[y1:y2, x1:x2]
                    cropped_img_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                    pil_cropped = Image.fromarray(cropped_img_rgb)
                    cropped_images.append(pil_cropped)
    return cropped_images
