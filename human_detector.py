#!/usr/bin/env python3
import logging
import os
import random
from argparse import ArgumentParser

import coloredlogs
import cv2
import numpy as np
import matplotlib.pyplot as plt
import dlib

LOCATION_MODEL = 'hog'
try:
    import dlib.cuda as cuda

    if cuda.get_num_devices() > 0 and dlib.DLIB_USE_CUDA:
        LOCATION_MODEL = 'cnn'
except ImportError:
    pass

import face_recognition as fr

FACE_TOLERANCE = 0.6

logger = logging.getLogger()


def load_types(basedir, human_prob):
    type_names = []
    type_imgs = {}
    for fname in os.listdir(basedir):
        full_path = os.path.join(basedir, fname)
        type_name = os.path.splitext(fname)[0]
        type_names.append(type_name)
        type_imgs[type_name] = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
    type_names = np.array(type_names)
    rest_prob = (1. - human_prob) / (len(type_names) - 1)
    type_probs = np.repeat(rest_prob, len(type_names))
    type_probs[type_names == 'human'] = human_prob
    max_len = max([len(s) for s in type_names])
    types = np.array(list(zip(type_names, np.cumsum(type_probs))), dtype=[('type', f'U{max_len}'), ('prob', 'f')])
    return types, type_imgs


def capture_image(capture):
    ret, frame = capture.read()
    if not ret:
        raise Exception('Failed to acquire the frame')
    return frame


def overlay(src, dst, box):
    top, right, bottom, left = box
    width = right - left
    height = bottom - top
    dst_bg = dst[top:bottom, left:right]
    src_resized = cv2.resize(src, (width, height))

    alpha = (src_resized[:, :, 3] / 255.) * .6
    beta = 1 - alpha

    overlay = alpha[:, :, np.newaxis] * src_resized[:, :, 0:3]
    original = beta[:, :, np.newaxis] * dst_bg

    dst[top:bottom, left:right] = cv2.add(overlay, original)


def get_faces(frame, ratio):
    faces = []

    detection_frame = cv2.resize(frame, (0, 0), fx=ratio, fy=ratio)
    locations = fr.face_locations(detection_frame, model=LOCATION_MODEL)
    if locations:
        encodings = fr.face_encodings(detection_frame, locations)
    else:
        encodings = []

    for location, encoding in zip(locations, encodings):
        top, right, bottom, left = location
        top = int(top / ratio)
        right = int(right / ratio)
        bottom = int(bottom / ratio)
        left = int(left / ratio)
        face_img = frame[top:bottom, left:right]
        faces.append(((top, right, bottom, left), face_img, encoding))

    return faces


class FaceDatabase(object):
    def __init__(self, tolerance=FACE_TOLERANCE):
        self.__known_faces = np.empty((0, 128))
        self.__type_prob = []
        self.__tolerance = tolerance

    def __insert_new(self, encoding):
        idx = len(self.__type_prob)
        prob = random.random()
        self.__type_prob.append(prob)
        self.__known_faces = np.vstack([self.__known_faces, encoding])
        return idx, prob

    def __call__(self, encoding):
        if self.__known_faces.size == 0:
            return self.__insert_new(encoding)

        distance = np.linalg.norm(self.__known_faces - encoding, axis=1)
        matching = np.nonzero(distance < self.__tolerance)[0]
        if matching.size == 0:
            return self.__insert_new(encoding)

        return matching[0], self.__type_prob[matching[0]]


def detector_loop(capture, types, type_imgs):
    NSKIP = 2
    cmap = plt.get_cmap('Dark2')
    fdb = FaceDatabase()

    cv2.namedWindow('capture', 0)

    i = 0
    while True:
        # Capture
        frame = capture_image(capture)
        ratio = 320. / frame.shape[0]
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # Detect
        if i % NSKIP == 0:
            faces = get_faces(frame, ratio)
            face_ids = []
            for _, _, encoding in faces:
                fi, fprob = fdb(encoding)
                face_ids.append((fi, fprob))

        for ((top, right, bottom, left), face_img, encoding), (face_id, face_prob) in zip(faces, face_ids):
            color = np.array(cmap(face_id % 8)) * 255
            color = color[[2, 1, 0, 3]]  # Transform to OpenCV BGRA
            color[3] = 0.  # 0 = fully opaque in OpenCV
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 5)
            label = types['type'][np.where(face_prob <= types['prob'])[0][0]]
            cv2.addText(
                display_frame, f'ID: {face_id} {label} ({face_prob:.2f})', (left, top - 12), 'monospace', color=color,
                weight=75
            )

        # Display
        cv2.imshow('capture', display_frame)
        # Events and refresh
        cv2.waitKey(10)
        if cv2.getWindowProperty('capture', cv2.WND_PROP_VISIBLE) < 1:
            break


parser = ArgumentParser()
parser.add_argument(
    '-c', '--camera', type=int, default=None,
    help='Capture device'
)
parser.add_argument(
    '-v', '--video', type=str,
    help='Capture from video'
)
parser.add_argument(
    '--types', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Types')
)
parser.add_argument(
    '--human', type=float, default=0.6, help='Human probability'
)

args = parser.parse_args()
if args.camera is None and args.video is None:
    parser.error('Need a capture source')
if args.camera is not None and args.video is not None:
    parser.error('Only one capture source supported')

coloredlogs.install(logging.DEBUG)

logger.info(f'Using model {LOCATION_MODEL}')

logger.info('Initializing video capture')
logger.debug(args.camera)
logger.debug(args.video)
capture = cv2.VideoCapture(args.camera if args.camera is not None else args.video)
if not capture.isOpened():
    capture.open(args.camera if args.camera else args.video)

logger.info(f'Loading type archive {args.types}')
types, types_imgs = load_types(args.types, args.human)

try:
    detector_loop(capture, types, types_imgs)
finally:
    logger.info('Destroying windows')
    cv2.destroyAllWindows()
