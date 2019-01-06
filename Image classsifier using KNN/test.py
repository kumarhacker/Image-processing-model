import os.path
import pickle
import face_recognition
import cv2




def predict(X_img, knn_clf=None, distance_threshold=0.5):
   
    X_face_locations = face_recognition.face_locations(X_img,number_of_times_to_upsample=3, model="hog")

  
    if len(X_face_locations) == 0:
        return []

 
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)

   
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]





frame = cv2.imread('google.png')
  
small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

rgb_small_frame = small_frame[:, :, ::-1]


with open("train_model.clf", 'rb') as f:
	knn_clf1 = pickle.load(f)

predictions = predict(rgb_small_frame, knn_clf=knn_clf1)

face_names = []


for name, (top, right, bottom, left) in predictions:
    top *= 4
    right *= 4
    bottom *= 4
    left *= 4

    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

   
    cv2.rectangle(frame, (left, bottom + 20), (right, bottom), (0, 0, 255), -1)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left, bottom + 10), font, 1.0, (0, 255, 0), 1)
 
cv2.imshow('frame',frame)
   
cv2.waitKey(0)

cv2.destroyAllWindows()
