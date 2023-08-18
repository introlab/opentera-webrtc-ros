import argparse
import os

from tqdm import tqdm

import cv2
import numpy as np

import torch

from datasets import FaceDetectionWider, HeadDetectionOpenImages
from datasets.detection_transforms import resize_image
from metrics import AveragePrecisionMetric

IMAGE_SIZE = (640, 640)


def main():
    parser = argparse.ArgumentParser(description='Evaluate OpenCV Face Detector')
    parser.add_argument('--dataset_root', type=str, help='Choose the dataset root path', required=True)
    parser.add_argument('--dataset_type', choices=['wider_face', 'open_images_head'],
                        help='Choose the dataset type', required=True)
    parser.add_argument('--output_path', type=str, help='Choose the output path', required=True)

    parser.add_argument('--cascade_classifier_path', type=str, help='Choose the cascade classifier path', required=True)
    parser.add_argument('--face_width_scale', type=float, help='Choose the face width scale', default=1.0)
    parser.add_argument('--face_height_scale', type=float, help='Choose the face height scale', default=1.0)

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    dataset = _create_dataset(args.dataset_type, args.dataset_root)
    face_detector = cv2.CascadeClassifier(args.cascade_classifier_path)

    _evaluate(dataset, face_detector, args.output_path, args.face_width_scale, args.face_height_scale)


def _create_dataset(dataset_type, dataset_root):
    def transform(pil_image, bboxes):
        resized_pil_image, scale = resize_image(pil_image, IMAGE_SIZE)
        resized_cv_image = cv2.cvtColor(np.array(resized_pil_image), cv2.COLOR_RGB2BGR)
        return resized_cv_image, bboxes * scale

    if dataset_type == 'wider_face':
        return FaceDetectionWider(dataset_root, split='validation', transform=transform)
    elif dataset_type == 'open_images_head':
        return HeadDetectionOpenImages(dataset_root, split='testing', transform=transform)
    else:
        raise ValueError('Invalid dataset type')


def _evaluate(dataset, face_detector, output_path, face_width_scale, face_height_scale):
    ap25_metric = AveragePrecisionMetric(iou_threshold=0.25, confidence_threshold=0.0)
    ap50_metric = AveragePrecisionMetric(iou_threshold=0.5, confidence_threshold=0.0)
    ap75_metric = AveragePrecisionMetric(iou_threshold=0.75, confidence_threshold=0.0)
    ap90_metric = AveragePrecisionMetric(iou_threshold=0.90, confidence_threshold=0.0)

    for cv_image, bboxes in tqdm(dataset):
        faces, num_detections = face_detector.detectMultiScale2(cv_image,
                                                                minNeighbors=1,
                                                                flags=cv2.CASCADE_SCALE_IMAGE,
                                                                minSize=(IMAGE_SIZE[0] // 10, IMAGE_SIZE[1] // 10))

        if len(faces) == 0:
            predictions = torch.zeros(1, 0, 5)
        else:
            predictions = torch.zeros(1, faces.shape[0], 5)

            faces = faces.astype(float)
            w = faces[:, 2]
            h = faces[:, 3]
            cx = faces[:, 0] + w / 2
            cy = faces[:, 1] + h / 2
            w *= face_width_scale
            h *= face_height_scale

            predictions[0, :, 0] = torch.from_numpy(num_detections)  # confidence
            predictions[0, :, 1] = torch.from_numpy(cx - w / 2)  # tl_x
            predictions[0, :, 2] = torch.from_numpy(cy - h / 2)  # tl_y
            predictions[0, :, 3] = torch.from_numpy(cx + w / 2)  # br_x
            predictions[0, :, 4] = torch.from_numpy(cy + h / 2)  # br_y

        targets = [bboxes]
        ap25_metric.add(predictions, targets)
        ap50_metric.add(predictions, targets)
        ap75_metric.add(predictions, targets)
        ap90_metric.add(predictions, targets)

    print('\nTest : AP@25={}, AP@0.5={}, AP@0.75={}, AP@0.9={}'.format(ap25_metric.get_value(),
                                                                       ap50_metric.get_value(),
                                                                       ap75_metric.get_value(),
                                                                       ap90_metric.get_value()))
    ap25_metric.save_curve(output_path, suffix='_25')
    ap50_metric.save_curve(output_path, suffix='_50')
    ap75_metric.save_curve(output_path, suffix='_75')
    ap90_metric.save_curve(output_path, suffix='_90')


if __name__ == '__main__':
    main()
