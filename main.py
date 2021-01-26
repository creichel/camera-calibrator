import numpy as np
import cv2
# import glob

# Termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object Points
objp = np.zeros((7*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:7].T.reshape(-1,2)

# Array to store object points and image points from all images
obj_points = [] # 3d point in real world space
img_points = [] # 2d points in image plane.

# Array to store images
images = []

camera_index = 0
found = True
while found:
    cap = cv2.VideoCapture(camera_index)
    ret, frame = cap.read()

    if not ret:
        found = False
        break

    print("Camera found, ID: " + str(camera_index))
    camera_index += 1

gray = None

# 1. Capture images and chessboard object points
cap = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    chessboardCornerRet, corners = cv2.findChessboardCorners(gray, (9, 7), None)

    # If found, add object points, image points (after refining them)
    if chessboardCornerRet:
        obj_points.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(frame, (9, 7), corners2, chessboardCornerRet)
        cv2.putText(img, "Calibration images taken: " + str(len(img_points)), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('img', img)
        cv2.waitKey(500)

    # Display the resulting frame
    cv2.putText(gray, "Calibration images taken: " + str(len(img_points)), (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('img', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 3. Calculate camera matrix and distortion coefficients
ret, mtx, dist, rotation_vectors, translation_vectors = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print ("camera matrix: \n" + str(mtx))
print ("distortion coefficients: \n" + str(dist))
print ("rotation vectors: \n" + str(dist))
print ("translation vectors: \n" + str(dist))

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

h, w = frame.shape[:2]
new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

print("New camera matrix: \n" + str(new_camera_mtx))

# 4. Store Coefficients in file
np.savetxt("camera_matrix.camera", new_camera_mtx, delimiter=',')
np.savetxt("distortion_coefficients.camera", dist, delimiter=',')

# 5. Calculate the error
mean_error = 0
for i in range(len(obj_points)):
    imgpoints2, _ = cv2.projectPoints(obj_points[i], rotation_vectors[i], translation_vectors[i], mtx, dist)
    error = cv2.norm(img_points[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print("total error: ", mean_error/len(obj_points))

# 6. Show comparison
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # undistort
    dst = cv2.undistort(frame, mtx, dist, None, new_camera_mtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    cropped_frame = frame[y:y + h, x:x + w]

    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(cropped_frame, alpha, dst, beta, 0.0)

    cv2.imshow('img', dst)
    if cv2.waitKey(1) & 0xFF == ord('w'):
        break

# FINAL Close down and output coefficients
cap.release()
cv2.destroyAllWindows()
