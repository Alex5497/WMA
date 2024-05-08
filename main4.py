import cv2 as cv
import numpy as np
import os

cap = cv.VideoCapture('dane/mov.MOV')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
MIN_MATCH_COUNT = 440
sift = cv.SIFT.create()
bf = cv.BFMatcher()

folder_path = 'pliki/'
keypoints_and_descriptors = []
images = []

# Przetwarzanie obrazów w folderze
for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.resize(gray, None, fx=0.5, fy=0.5)  # Zmniejszenie rozmiaru obrazów
    print(filename)
    images.append(gray)
    kp, des = sift.detectAndCompute(gray, None)
    keypoints_and_descriptors.append((kp, des))

# Obliczanie punktów kluczowych i deskryptorów dla pierwszej klatki
ret, frame = cap.read()
gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# gray_frame = cv.resize(gray_frame, None, fx=0.5, fy=0.5)  # Zmniejszenie rozmiaru klatki
kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

out = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*'DIVX'), 30, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

    matches = [bf.match(des, des_frame) for _, des in keypoints_and_descriptors]

    best_matches_count = 0
    best_matched_image_index = 0
    good_matches = []

    # Obliczanie średniej odległości dla punktów dopasowania
    mean_distance = np.mean([m.distance for match in matches for m in match])

    for match in matches:
        very_good = [m for m in match if m.distance < 0.7 * mean_distance]  # Ograniczenie punktów dopasowania
        good_matches.append(very_good)

    for i, good_match in enumerate(good_matches):
        if len(good_match) > best_matches_count:
            best_matches_count = len(good_match)
            best_matched_image_index = i

    very_good = good_matches[best_matched_image_index]
    best_matched_image = images[best_matched_image_index]

    if len(very_good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints_and_descriptors[best_matched_image_index][0][m.queryIdx].pt for m in very_good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in very_good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = best_matched_image.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        frame = cv.polylines(frame, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
        print("Matches are found - {}/{}".format(len(very_good), MIN_MATCH_COUNT))
    else:
        print("Not enough matches are found - {}/{}".format(len(very_good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(255, 50, 200),
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img4 = cv.drawMatches(best_matched_image, keypoints_and_descriptors[best_matched_image_index][0], frame, kp_frame, very_good, None, **draw_params)
    out.write(img4)
    cv.imshow('frame', img4)
    if cv.waitKey(2) & 0xFF == ord('q'):
        break


cap.release()
cv.destroyAllWindows()
