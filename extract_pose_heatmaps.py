import mxnet as mx
from mxnet import gluon
from gluoncv import model_zoo, data
from gluoncv.data.transforms.pose import detector_to_simple_pose
import cv2

import os
import random
from pathlib import Path

from stair_safety_net import StairSafetyNet

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def mat_to_mxarray(img, context):
    np_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mx_array = mx.nd.array(np_array)
    input, orig = data.transforms.presets.ssd.transform_test(mx_array, 512)
    return input.as_in_context(context)


def get_data(path):
    labeled_paths = {}
    for root, dirs, files in os.walk(path):
        parent = Path(root).name
        action = parent
        for file in files:
            labeled_paths[os.path.join(path, parent, file)] = action
    return labeled_paths


def iterate_over_videos(labeled_videos, detector, pose_net, save_path, context):
    video_paths = list(labeled_videos.keys())
    for video_path in video_paths:
        print(video_path)
        video_name = Path(video_path).stem
        label = labeled_videos[video_path]
        video_heatmaps = mx.nd.empty((0, 17, 64, 48))
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            ret, img = cap.read()
            if not ret:
                break
            nd_array = mat_to_mxarray(img, context)
            class_IDs, scores, bounding_boxs = detector(nd_array)
            all_detections, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, ctx=context, thr=0.7)

            if all_detections is None: 
                continue
            detection = all_detections[0].expand_dims(0)

            heatmap = pose_net(detection)
            try:
                video_heatmaps = mx.nd.concat(video_heatmaps, heatmap, dim=0)
            except mx.base.MXNetError:
                video_heatmaps = heatmap

        mx.nd.save(os.path.join(save_path, label, video_name + ".mat"), video_heatmaps)



def extract_heatmaps():
    model_json = 'model/Ultralight-Nano-SimplePose.json'
    model_params = "model/Ultralight-Nano-SimplePose.params"
    TRAIN_SET = "data/train"
    TEST_SET = "data/test"

    device = mx.gpu()
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, ctx=device)
    detector.reset_class(["person"], reuse_weights=['person'])
    pose_net = gluon.SymbolBlock.imports(model_json, ['data'], model_params, ctx=device)

    train_videos = get_data(TRAIN_SET)
    test_videos = get_data(TEST_SET)

    iterate_over_videos(train_videos, detector, pose_net, "ndarray-data/train", device)
    iterate_over_videos(test_videos, detector, pose_net, "ndarray-data/test", device)
        


if __name__ == "__main__":
    extract_heatmaps()