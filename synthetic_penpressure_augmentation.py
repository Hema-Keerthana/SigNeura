import os
import cv2
import numpy as np

INPUT_DATASET = r"D:\Project\datasets"
OUTPUT_DATASET = r"D:\Project\augmented_pressure_datasets"
os.makedirs(OUTPUT_DATASET, exist_ok=True)

def generate_pressure_map(img):
    _, binary = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)
    return dist

def apply_pressure(img, pressure_map):
    img = img.astype(np.float32) / 255.0
    alpha = np.random.uniform(0.6, 1.4)
    pressured = img * (1 + alpha * pressure_map)
    pressured = np.clip(pressured, 0, 1.0)
    return (pressured * 255).astype(np.uint8)

def stroke_variation(img):
    k = np.random.choice([1, 2, 3])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(img, kernel) if np.random.rand() > 0.5 else cv2.erode(img, kernel)

for person in os.listdir(INPUT_DATASET):
    person_path = os.path.join(INPUT_DATASET, person)
    if not os.path.isdir(person_path):
        continue

    out_person = os.path.join(OUTPUT_DATASET, person)
    os.makedirs(out_person, exist_ok=True)

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        cv2.imwrite(os.path.join(out_person, img_name), img)

        for i in range(4):
            pm = generate_pressure_map(img)
            pimg = apply_pressure(img, pm)
            pimg = stroke_variation(pimg)
            base, ext = os.path.splitext(img_name)
            cv2.imwrite(
                os.path.join(out_person, f"{base}_pressure{i+1}{ext}"),
                pimg
            )

print("✅ Final pressure augmentation completed")
