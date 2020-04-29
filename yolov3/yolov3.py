import os
import time
import cv2
import numpy as np
import argparse
from sys import platform


from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *


class Yolov3Detector:
    """
    Initialization of detector API using pretrained model
    Creates Graph and initializes it with tensorflow checkpoint,
    and subsequently create session object based on Graph
    Model is YOLOv3
    """
    def __init__(self):
        #class params
        #self.yolo_cfg = 'yolov3/cfg/yolov3-tiny.cfg'
        #self.weights = 'yolov3/weights/yolov3-tiny.pt'

        self.yolo_cfg = 'yolov3/cfg/yolov3-spp.cfg'
        self.weights = 'yolov3/weights/yolov3-spp-ultralytics.pt'

        self.webcam = False 
        self.img_size = 512 #320x512 is default img size

        self.device = torch_utils.select_device()

        # Initialize model
        self.model = Darknet(self.yolo_cfg, self.img_size)

        # Load weights
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)
        
        #set model to evaluation mode
        self.model.to(self.device).eval()

        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference

        self.classes = load_classes('yolov3/data/coco.names')

        _ = self.model(torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)) if self.device.type != 'cpu' else None  # run once


    def getBoundingBoxes(self, image):

        conf_thres = 0.3
        iou_thres = 0.6

        t0 = time.time()

        im0s = image.copy()
        img = letterbox(im0s, new_shape=self.img_size)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        # Process for tensorflow
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=False)[0]
        t2 = torch_utils.time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres,
                                   multi_label=False, classes=None, agnostic=False)

        # Process detections
        boundingBoxes = []
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.classes[int(c)])  # add to string

                # Store results in array
                for *xyxy, conf, detected_cls in det:
                    coords = np.zeros(4)
                    if detected_cls == 0:
                        for num, tensor in enumerate(xyxy):
                                coords[num] = tensor.data
                        boundingBoxes.append(coords)
            # Print time (inference + NMS)

            print('%sDone. (%.3fs)' % (s, t2 - t1))
        return boundingBoxes

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/alson_fam.mp4', help='source')  # input file/folder, 0 for webcam
    opt = parser.parse_args()


    detector = Yolov3Detector() 
    print(opt)

    if opt.source == '0':
        cap = cv2.VideoCapture(0)
    elif opt.source:
        cap = cv2.VideoCapture(opt.source)
    else: 
        cap = cv2.VideoCapture('./../videos/alson_fam.mp4')

    #save video file
    #out = cv2.VideoWriter('./output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (1280,720))
    
    while(True):
        t1 = time.time()
        r, img = cap.read()

        boxes_coords = detector.getBoundingBoxes(img)

        for i in range(len(boxes_coords)):
            # Class 1 represents human
                box = boxes_coords[i]
                box = box.astype(int)
                # draw the bounding box on the image
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                #print("Object detected i=%s ####, box1=%s box0=%s box3=%s box2=%s" %(str(i), str(box[1]), str(box[0]),str(box[3]),str(box[2])))

        fps = int(1/(time.time() - t1))
        print("FPS: %d" % fps)
        cv2.imshow("person re-id", img)
        #out.write(img)

        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break