#!/usr/bin/env python2
#
# Example to compare the faces in two images.
# Brandon Amos
# 2015/09/29
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
#import itertools
import os
import shutil
import matplotlib.pyplot as pl
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

import numpy as np
np.set_printoptions(precision=2)

import openface

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

parser = argparse.ArgumentParser()

parser.add_argument('imgs', type=str, nargs='+', help="Input images.")
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                    default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--saveboxes', action='store_true')

args = parser.parse_args()

if args.verbose:
    print("Argument parsing and loading libraries took {} seconds.".format(
        time.time() - start))

start = time.time()
align = openface.AlignDlib(args.dlibFacePredictor)
net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
if args.verbose:
    print("Loading the dlib and OpenFace models took {} seconds.".format(
        time.time() - start))

if os.path.isdir("./output"):
    shutil.rmtree("./output")
os.makedirs("./output")

imgIndex = 0

def getReps(imgPath):
    if args.verbose:
        print("Processing {}.".format(imgPath))
    name = os.path.basename(imgPath)
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    start = time.time()
    boxes = align.getAllFaceBoundingBoxes(rgbImg)
    if args.verbose:
        print("  + Detected faces count: {} from file: {}".format(len(boxes), name))
        print("  + Face detection took: {} seconds.".format(time.time() - start))    

    if len(boxes) == 0:
        print("  + No faces from file: {}".format(name))
        return [], [], name

    if args.saveboxes:
        boxedImgPath = "./output/" + os.path.splitext(name)[0] + "_boxed.png" 
        shutil.copy(imgPath, boxedImgPath)
        boxedImg = cv2.imread(boxedImgPath)
        boxedImg = cv2.cvtColor(boxedImg, cv2.COLOR_BGR2RGB)
        for i in range(0, len(boxes)):
            box = boxes[i]
            cv2.rectangle(boxedImg, (box.left(), box.top()), (box.right(), box.bottom()), (0,255,0), 3)
            pl.imsave(boxedImgPath, boxedImg)

    start = time.time()
    faces = []
    global imgIndex
    for i in range(0, len(boxes)):
        face = align.align(args.imgDim, rgbImg, boxes[i], landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        pl.imsave("./output/" + str(imgIndex) + ".png", face)
        faces.append(face)
        imgIndex = imgIndex + 1

    if args.verbose:
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    start = time.time()
    reps = []
    for i in range(0, len(faces)):
        rep = net.forward(faces[i])
        reps.append(rep)

    if args.verbose:
        print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
        for i in range(0, len(reps)):
            print("Representation#{}:".format(i))
            print(reps[i])
            print("=====\n")
        print("-----\n")
    return reps, boxes, name

allReps = []
allNames = []
allImages = []
for img in args.imgs:
    reps, boxes, name = getReps(img)
    for i in range(0, len(reps)):
        allReps.append(reps[i])
        allNames.append(name)
        allImages.append(img)

if args.verbose:
    print("Result reps array length: {}".format(len(allReps)))
    print("Result reps array: {}".format(allReps))
    print("Result names array: {}".format(allNames))

if args.verbose:
    dist = euclidean_distances(allReps)
    print("Euclidian distance: {}".format(dist))

#allRepsNorm = StandardScaler().fit_transform(allReps)
#if args.verbose:
#    print("Result reps array: {}", allRepsNorm)

db = DBSCAN(eps=0.9, min_samples=1).fit(allReps)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

if args.verbose:
    print("Detected number of clusters: {}".format(n_clusters_))
    print("Detected lables: {}".format(labels))

for i in range(0, len(labels)):
    if labels[i] < 0:
        continue
    clusterName = str(labels[i])
    previewFileName = str(i) + ".png"
    fileName = allNames[i]
    clusterFolder = "./output/" + clusterName + "/"
    previewFilePath = clusterFolder + previewFileName
    if not os.path.exists(clusterFolder):
        os.makedirs(clusterFolder)
    if not os.path.exists(clusterFolder + fileName):
        shutil.copy(allImages[i], clusterFolder + fileName)
    if not os.path.exists(previewFilePath):
        shutil.copy("./output/" + previewFileName, previewFilePath)


