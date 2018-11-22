import logging
import os
from argparse import ArgumentParser

import coloredlogs
import cv2
import numpy as np

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
    dst_bg = dst[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
    src_resized = cv2.resize(src, dst_bg.shape[0:2])

    alpha = (src_resized[:, :, 3] / 255.) * .6
    beta = 1 - alpha

    overlay = alpha[:, :, np.newaxis] * src_resized[:, :, 0:3]
    original = beta[:, :, np.newaxis] * dst_bg

    dst[box[1]:box[1] + box[3], box[0]:box[0] + box[2]] = cv2.add(overlay, original)


def detector_loop(capture, classifier, face_recon, types, type_imgs):
    recon_shape = face_recon.getMean().shape
    while True:
        # Capture
        frame = capture_image(capture)
        frame = cv2.resize(frame, (640, 480))
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        # Detect
        faces = classifier.detectMultiScale(frame_gray, 1.1, 10, flags=cv2.CASCADE_SCALE_IMAGE)
        for face in faces:
            cv2.rectangle(frame, (face[0], face[1]), (face[0] + face[2], face[1] + face[3]), (0., 0., 255.), 5)
            face_img_gray = frame_gray[face[1]:face[1] + face[3], face[0]:face[0] + face[2]]
            face_resized = cv2.resize(face_img_gray, recon_shape)
            label, _ = face_recon.predict(face_resized)
            matching = types['type'][np.where(label <= types['prob'])[0][0]]
            matching_img = types_imgs[matching]
            overlay(matching_img, frame, face)

        # Display
        cv2.imshow('capture', frame)
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
    '--cascade', type=str, default='/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml',
    help='Cascade XML'
)
parser.add_argument(
    '--fisher', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'yale.yml'),
    help='Pre-trained Fisher model'
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

logger.info('Initializing video capture')
logger.debug(args.camera)
logger.debug(args.video)
capture = cv2.VideoCapture(args.camera if args.camera is not None else args.video)
if not capture.isOpened():
    capture.open(args.camera if args.camera else args.video)

logger.info(f'Loading cascade classifier {args.cascade}')
classifier = cv2.CascadeClassifier(args.cascade)

logger.info(f'Loading Fisher model {args.fisher}')
face_recon = cv2.face.FisherFaceRecognizer_create(80)
face_recon.read(args.fisher)

logger.info(f'Loading type archive {args.types}')
types, types_imgs = load_types(args.types, args.human)
types['prob'] *= face_recon.getLabels().max()

try:
    detector_loop(capture, classifier, face_recon, types, types_imgs)
finally:
    logger.info('Destroying windows')
    cv2.destroyAllWindows()
