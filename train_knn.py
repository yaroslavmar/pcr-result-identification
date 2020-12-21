# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os


def image_to_feature_vector(image, size=(300, 300)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()


def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram if we are using OpenCV 2.4.X
	if imutils.is_cv2():
		hist = cv2.normalize(hist)
	# otherwise, perform "in place" normalization in OpenCV 3 (I
	# personally hate the way this is done
	else:
		cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()


# grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images('./data/data1/train/'))


#os.listdir('./data/data1/train/')
#list_files("./data/data1/train/", validExts=image_types)
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

# loop over the input images



for (i, imagePath) in enumerate(imagePaths):

	# load the image and extract the class label (assuming that our
	# path as the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-2].split(".")[0]
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
	# update the raw images, features, and labels matricies,
	# respectively
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))


rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))


(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=666)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=666)



train_features = features[:-2]
test_features = features[-2:]
train_labels = labels[:-2]
test_labels = labels[-2:]

print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=1,
	n_jobs=1)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))


# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=1,
	n_jobs=1)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))


# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=1,
	n_jobs=1)
model.fit(train_features, train_labels)
acc = model.score(test_features, test_labels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

model.predict_proba(test_features)


from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(rawImages)


model = KNeighborsClassifier(n_neighbors=1,
	n_jobs=1)
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)

list_acc = []
for train_index, test_index in loo.split(rawImages):
	X_train, X_test = rawImages[train_index], rawImages[test_index]
	y_train, y_test = labels[train_index], labels[test_index]
	model = KNeighborsClassifier(n_neighbors=1,n_jobs=1)
	model.fit(X_train, y_train)
	acc = model.score(X_test, y_test)
	list_acc.append(acc)
