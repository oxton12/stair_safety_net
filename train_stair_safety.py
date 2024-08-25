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


def get_data(path, context):
    labeled_paths = {}
    for root, dirs, files in os.walk(path):
        parent = Path(root).name
        action = mx.nd.array([0, 1]) if parent == "not_holding" else mx.nd.array([1, 0])
        mx_action = mx.nd.array(action).as_in_context(context)
        for file in files:
            labeled_paths[os.path.join(path, parent, file)] = mx_action
    return labeled_paths


def evaluate_test(test_videos, detector, net):
    L2 = gluon.loss.L2Loss()
    accuracy = mx.metric.Accuracy()
    for video_path in test_videos.keys():
        label = test_videos[video_path]
        cap = cv2.VideoCapture(video_path)
        while(cap.isOpened()):
            loss = 0
            ret, img = cap.read()
            if not ret:
                break
            nd_array = mat_to_mxarray(img)
            class_IDs, scores, bounding_boxs = detector(nd_array)
            detection, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
            output = net(detection)
            if output is not None:
                loss += L2(output, label)
                accuracy.update(label, output)
    _, acc = accuracy.get()
    return acc, loss


def train():
    TRAIN_SET = "data/train"
    TEST_SET = "data/test"
    EPOCHS = 5
    lr = 0.01
    LSTM_layer_size = 32
    LSTM_layeres = 2
    sequence_len = 4
    log_interval = 10

    device = mx.gpu()
    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True, ctx=device)
    detector.reset_class(["person"], reuse_weights=['person'])
    net = StairSafetyNet(LSTM_layer_size, LSTM_layeres, sequence_len, context=device)

    train_videos = get_data(TRAIN_SET, device)
    test_videos = get_data(TEST_SET, device)

    net.initialize(mx.init.Xavier(), ctx=device)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
    L2 = gluon.loss.L2Loss()
    accuracy = mx.metric.Accuracy()

    for epoch in range(EPOCHS):
        accuracy.reset()
        output_count = 0
        video_paths = list(train_videos.keys())
        random.shuffle(video_paths)
        for video_path in video_paths:
            label = train_videos[video_path]
            stacked_detection = mx.nd.empty((0, 3, 256, 192), ctx=device)
            cap = cv2.VideoCapture(video_path)
            while(cap.isOpened()):
                ret, img = cap.read()
                if not ret:
                    break
                nd_array = mat_to_mxarray(img, device)
                class_IDs, scores, bounding_boxs = detector(nd_array)
                all_detections, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs, ctx=device, thr=0.7)
                if all_detections is None: 
                    continue
                detection = all_detections[0].expand_dims(0)
                if stacked_detection.shape[0] == 0:
                    stacked_detection = detection
                    continue
                stacked_detection = mx.nd.concat(stacked_detection, detection, dim=0)
                if stacked_detection.shape[0] != 4:
                    continue
                with mx.autograd.record():
                    output = net.hybrid_forward(stacked_detection)
                    loss = L2(output, label)
                mx.autograd.backward(loss, retain_graph=True)
                trainer.step(1)
                accuracy.update(label, output)
                stacked_detection = stacked_detection[1:]
                output_count += 1

                if output_count % log_interval == 0:
                    _, acc = accuracy.get()
                    print(f"""Epoch[{epoch + 1}] Batch[{output_count}] loss = {loss} | accuracy = {acc}""")

        _, acc = accuracy.get()
        test_acc, test_loss = evaluate_test(test_videos, detector, net)
        print(f"[Epoch {epoch + 1}] training: accuracy={acc} | validation loss={test_loss} | validation accuracy={test_acc}")

    net.save("stair_safety")


if __name__ == "__main__":
    train()