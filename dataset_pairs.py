import os
import cv2
import numpy as np
from sklearn.utils import shuffle

def preprocess_image(path, size=(100, 100)):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

def create_signature_pairs(data_dir):
    X1, X2, y = [], [], []
    persons = sorted(os.listdir(data_dir))
    person_genuine = {}

    for person in persons:
        p_path = os.path.join(data_dir, person)
        if not os.path.isdir(p_path):
            continue

        files = os.listdir(p_path)
        genuine_files = [f for f in files if "genuine" in f.lower()]
        forged_files  = [f for f in files if "forged"  in f.lower()]

        genuine_imgs = [preprocess_image(os.path.join(p_path, f)) for f in genuine_files]
        forged_imgs  = [preprocess_image(os.path.join(p_path, f)) for f in forged_files]

        person_genuine[person] = genuine_imgs

        # Genuine–Genuine (positive)
        for i in range(len(genuine_imgs)):
            for j in range(i + 1, len(genuine_imgs)):
                X1.append(genuine_imgs[i])
                X2.append(genuine_imgs[j])
                y.append(1)

        # Genuine–Forged (negative)
        for g in genuine_imgs:
            for f in forged_imgs:
                X1.append(g)
                X2.append(f)
                y.append(0)

    # Cross-person Genuine–Genuine (negative)
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            for img1 in person_genuine[persons[i]]:
                for img2 in person_genuine[persons[j]]:
                    X1.append(img1)
                    X2.append(img2)
                    y.append(0)

    X1, X2, y = shuffle(
        np.array(X1), np.array(X2), np.array(y),
        random_state=42
    )

    print("Total pairs:", len(y))
    return (X1, X2), y
