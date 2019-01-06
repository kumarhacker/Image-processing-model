
import math
from sklearn import neighbors
import os
import os.path
import pickle

import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import cv2

import time
import glob

def train(model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    X = []
    y = []

    for class_dir in os.listdir("images/"):

        if not os.path.isdir(os.path.join("images/", class_dir)):
            continue

        for img_path in image_files_in_folder(os.path.join("images", class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:

                zz=os.listdir("images/"+class_dir)
                if zz[0].endswith(".jpg"):

                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])

                    y.append(class_dir)






    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)


    with open("train_model.clf", 'wb') as f:
        pickle.dump(knn_clf, f)

train()
