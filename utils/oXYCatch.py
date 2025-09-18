# YOLOv8 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv8 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov8s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov8s.pt                 # PyTorch
                                 yolov8s.torchscript        # TorchScript
                                 yolov8s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov8s_openvino_model     # OpenVINO
                                 yolov8s.engine             # TensorRT
                                 yolov8s.mlmodel            # CoreML (macOS-only)
                                 yolov8s_saved_model        # TensorFlow SavedModel
                                 yolov8s.pb                 # TensorFlow GraphDef
                                 yolov8s.tflite             # TensorFlow Lite
                                 yolov8s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov8s_paddle_model       # PaddlePaddle
"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path
import yaml
import numpy as np
import torch
import shutil
import datetime

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
print(ROOT)

from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from ultralytics import YOLO
import uvicorn
import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
logging.basicConfig(filename='example.log', level=logging.DEBUG,format='\r\n%(asctime)s %(levelname)s：%(message)s')
# 打开日志文件并将其截断为零字节
with open('example.log', 'w'):
    pass

# 上次获取的时间
last_time = datetime.datetime(1, 1, 1, 0, 0, 0)  # 创建一个时间为 0 的 datetime 对象

class Yolo():
    def __init__(self,opt):
        self.weights =  opt['runH5File']
        self.device = '0' if (opt['gpuOpenOrClose']) else 'cpu' # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.nosave = False
        self.model = None
        self.data = ROOT / 'data/618A.yaml' # dataset.yaml path
        self.imgsz = [640] # inference size (height, width)
        self.conf = opt['identifyScore'] # confidence threshold
        self.iou = opt['identifyIou']  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.view_img = True # show results
        self.save_txt = True  # save results to *.txt
        self.save_conf = True  # save confidences in --save-txt labels
        self.save_crop = True  # save cropped prediction boxes
        self.nosave = False  # do not save images/videos
        # classes=None,  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False # visualize features
        # update=False,  # update all models
        self.project = ROOT / 'runs/detect'  # save results to project/name
        self.name = 'exp'  # save results to project/name
        self.exist_ok = True  # existing project/name ok, do not increment
        self.line_thickness = 2  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.vid_stride = 1  # video frame-rate stride
        self.retina_masks = False  # High resolution masks
        self.board_border = opt['board_border']
        self.userId = opt['userId']

    def run(self):
        '''预热模型'''
        # 检查是否有可用的GPU
        if torch.cuda.is_available():
            print('GPU is available')
        else:
            print('No GPU available')
            self.device = 'cpu'
        # Load model
        select_device(self.device)
        self.model = YOLO(self.weights)
        if not self.model:
            print('未找到模型，请检查模型是否存在')
            logging.error("【未找到模型，请检查模型是否存在】")
            sys.exit()

        # model.predict(source=source, save=not nosave,iou=iou, conf=conf, save_txt=save_txt, save_conf=save_conf,
        #               save_crop=save_crop, agnostic_nms=agnostic_nms, augment=augment,
        #               visualize=visualize, project=project, name=name, exist_ok=exist_ok,
        #               line_thickness=line_thickness, hide_labels=hide_labels, hide_conf=hide_conf, half=half, dnn=dnn,
        #               vid_stride=vid_stride, retina_masks=retina_masks)
    def predict(self,category,flag=True):
        # with open(self.data, 'w', encoding='utf-8') as f:
        #     pass
        dst_file = 'labels/' + category + '.txt'
        imgfile = category + '.jpg'
        sourcefile = 'img/input/' + imgfile
        self.FileSetLimits()
        self.source = str(ROOT / sourcefile)
        print(self.source)
        # self.source = str(self.source)
        save_img = not self.nosave and not self.source.endswith('.txt')  # save inference images
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = self.source.isnumeric() or self.source.endswith('.streams') or (is_url and not is_file)
        screenshot = self.source.lower().startswith('screen')
        if is_url and is_file:
            self.source = check_file(self.source)  # download
        # Directories
        self.save_dir = increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok)  # increment run
        # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        with open(self.save_dir / dst_file, 'w', encoding='utf-8') as f:
            pass

        if not is_url and is_file:
            img = cv2.imread(self.source)
            height, width, channels = img.shape
            if width != 848:
                print('没有新图片传入！')
                logging.error("【没有新图片传入】")
                return
            img_wide = (width - 854) / 2
            img = img[:, int(106 + img_wide):int(746 + img_wide)]
            height, width, channels = img.shape
            cv2.imwrite(self.source, img)
        else:
            print(self.source)
            cap = cv2.VideoCapture(self.source)
            ret, frame = cap.read()
            if ret:
                img = frame
            else:
                print('摄像头有问题')
                logging.error("【摄像头有问题】")
                return
        print(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))
        self.model.predict(source=self.source, save=not self.nosave, iou=self.iou, conf=self.conf, save_txt=self.save_txt, save_conf=self.save_conf,
                      save_crop=self.save_crop, agnostic_nms=self.agnostic_nms, augment=self.augment,
                      visualize=self.visualize, project=self.project, name=self.name, exist_ok=self.exist_ok,
                      line_thickness=self.line_thickness, hide_labels=self.hide_labels, hide_conf=self.hide_conf, half=self.half, dnn=self.dnn,
                      vid_stride=self.vid_stride, retina_masks=self.retina_masks)
        print(increment_path(Path(self.project) / self.name, exist_ok=self.exist_ok))
        '''将识别框的中心点像素坐标写入数据库'''
        # r = redis.Redis(host="127.0.0.1", port=6379)
        with open(self.data, 'r', encoding='utf-8') as f:
            self.data_yaml = yaml.load(f, Loader=yaml.FullLoader)

        class_names = [] # 标签类别
        for value in self.data_yaml['names'].values():
            class_names.append(value)

        labellist = [] # 返回值
        for thislebal in class_names:
            labellist.append([thislebal, []])

        recognizes = [] # 识别到的类别
        if self.save_txt:
            with open(self.save_dir / dst_file,'r',encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                words = line.strip().split()
                recognizes.append([self.data_yaml['names'][int(words[0])],words[1:]])

            outpath = 'img/output/' + imgfile
            print(outpath)
            shutil.copy(self.save_dir / imgfile, ROOT / outpath)

        for label in labellist:
            for recognize in recognizes:
                if recognize[0] == label[0]:
                    x, y = recognize[1][0], recognize[1][1]
                    x_center = float(x) * width
                    y_center = float(y) * height
                    if not flag:
                        if label[0] == 'transparentempty':
                            label[1].append([round(y_center,1),round(x_center,1),1,round(float(recognize[1][-1]),1),'transparentempty'])
                        elif label[0] == 'transparenexcept':
                            label[1].append([round(y_center,1),round(x_center,1),1,round(float(recognize[1][-1]),1),'transparenexcept'])
                        elif label[0] == 'withoutcover':
                            label[1].append([round(y_center,1),round(x_center,1),1,round(float(recognize[1][-1]),1),'withoutcover'])
                        elif label[0] == 'havecover':
                            label[1].append([round(y_center,1),round(x_center,1),1,round(float(recognize[1][-1]),1),'havecover'])
                        elif label[0] == 'none':
                            label[1].append([round(y_center,1),round(x_center,1),1,round(float(recognize[1][-1]),1),'none'])
                        else:
                            label[1].append([round(y_center, 1), round(x_center, 1), 1, round(float(recognize[1][-1]), 1)])
                    else:
                        label[1].append([round(y_center,1),round(x_center,1),1,round(float(recognize[1][-1]),1)])
        return labellist

    def model1vadiocatch(self,getnewOxy, k):
        '''
        识别颗粒
        '''
        try:
            board_border_1 = self.board_border.split(',')
            board_border = [(int(board_border_1[0]),int(board_border_1[1])),(int(board_border_1[2]),int(board_border_1[3])),(int(board_border_1[4]),int(board_border_1[5])),(int(board_border_1[6]),int(board_border_1[7]))]
            print('+++++',board_border)
            labellist = self.predict('particle')
            res = []
            print(getnewOxy, k)
            for thislabel in labellist:
                labelxy = [0, 0, 0, -1]
                for thislabelist in thislabel[1]:
                    print((thislabelist[1], thislabelist[0]),board_border)
                    if self.IsPointInMatrix((thislabelist[1], thislabelist[0]), board_border):
                        bestscore = thislabelist[3]
                        labelxy[:3] = (k * getnewOxy * np.asmatrix(thislabelist[:3]).T)
                        labelxy[3] = bestscore
                res.append([float(labelxy[0]), float(labelxy[1]), labelxy[3], len(thislabel[1]), thislabel[0]])
            if (datetime.datetime.now() - last_time).total_seconds() > 60:  # 当前请求时间比上次请求时间多60秒，则判断为新一轮
                Lv = len(labellist[2][1]) # 测试用
            else:
                Lv = len(labellist[2][1]) # 测试用
                # Lv = '-1'
            res.append(Lv)
            print(res)
            return res
        except Exception as e:
            logging.exception("【识别颗粒出错】")
            print(e)
    def model2vadiocatch(self,getnewOxy, k):
        '''
        识别托盘
        '''
        try:
            labellist = self.predict('tray',flag=False)
            if not labellist:
                return {'resflag': -1}
            transparentemptylist = labellist[self.get_key(self.data_yaml['names'], 'transparentempty')][1]
            transparenexceptlist = labellist[self.get_key(self.data_yaml['names'], 'transparenexcept')][1]
            withoutcoverlist = labellist[self.get_key(self.data_yaml['names'], 'withoutcover')][1]
            havecoverlist = labellist[self.get_key(self.data_yaml['names'], 'havecover')][1]
            nonelist =  labellist[self.get_key(self.data_yaml['names'], 'none')][1]
            XLabellist = labellist[self.get_key(self.data_yaml['names'], 'XLabel')][1]
            YLabellist = labellist[self.get_key(self.data_yaml['names'], 'YLabel')][1]
            print('______________________', transparentemptylist, transparenexceptlist, withoutcoverlist, havecoverlist,
                  nonelist, XLabellist, YLabellist)
            XLabelxy = [0, 0, 0]
            Ylabelxy = [0, 0, 0]
            res = {
                'resflag': 1,
                'XLabel': [0.0, 0.0, ""],
                'YLabel': [0.0, 0.0, ""]
            }
            res = {
                'resflag': 1,
                'XLabel': {"X": 0.0, "Y": 0.0, "strType": ""},
                'YLabel': {"X": 0.0, "Y": 0.0, "strType": ""}
            }
            reslabel = [transparentemptylist, transparenexceptlist, withoutcoverlist, havecoverlist, nonelist]
            if len(XLabellist) == 1 and len(YLabellist) == 1:
                XLabelxy = (k * getnewOxy * np.asmatrix(XLabellist[0][:3]).T)
                YLabelxy = (k * getnewOxy * np.asmatrix(YLabellist[0][:3]).T)
                res['XLabel']["X"] = float(XLabelxy[0])
                res['XLabel']["Y"] = float(XLabelxy[1])
                res['YLabel']["X"] = float(YLabelxy[0])
                res['YLabel']["Y"] = float(YLabelxy[1])
                newreslabel = [i for i in reslabel if len(i) > 0]
                if (len(newreslabel) == 1):
                    res['XLabel']["strType"] = newreslabel[0][0][4]
                    res['YLabel']["strType"] = newreslabel[0][1][4]
                elif (len(newreslabel) == 2):
                    if (CompareVectorDistance(XLabellist[0][:2], YLabellist[0][:2], newreslabel[0][0][:2],
                                              newreslabel[1][0][:2])):
                        res['XLabel']["strType"] = newreslabel[0][0][4]
                        res['YLabel']["strType"] = newreslabel[1][0][4]
                    else:
                        res['XLabel']["strType"] = newreslabel[1][0][4]
                        res['YLabel']["strType"] = newreslabel[0][0][4]
            else:
                res = {'resflag': -1}
            return res
        except Exception as e:
            print(e)
            logging.exception("【识别托盘出错】")
            return {'resflag': -1}

    def FileSetLimits(self,path=ROOT / 'runs/detect',num=100):
        '''文件超过上限数量则清空'''
        if len([f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]) > num:
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    if os.path.isdir(file_path):
                        # os.rmdir(file_path)
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
    def get_key(self,d, value):
        '''
        值对应的键
        '''
        for k, v in d.items():
            if v == value:
                return k
        return None

    def compute_iou(self,rec1, rec2):
        """
        求两框的iou
        rec1: (x0, y0, x1, y1)
        rec2: (x0, y0, x1, y1)
        :return: scala value of IoU
        """
        # computing area of each rectangle
        S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

        # computing the sum_area
        sum_area = S_rec1 + S_rec2

        # find the each edge of intersect rectangle
        left_line = max(rec1[1], rec2[1])
        right_line = min(rec1[3], rec2[3])
        top_line = max(rec1[0], rec2[0])
        bottom_line = min(rec1[2], rec2[2])
        # print(top_line, left_line, right_line, bottom_line)

        # judge if there is an intersect area
        if left_line >= right_line or top_line >= bottom_line:
            return 0
        else:
            intersect = (right_line - left_line) * (bottom_line - top_line)
            return (intersect / (sum_area - intersect)) * 1.0

    def GetCross(self,p1, p2, p):
        p1_x, p1_y = p1
        p2_x, p2_y = p2
        p_x, p_y = p
        return (p2_x - p1_x) * (p_y - p1_y) - (p_x - p1_x) * (p2_y - p1_y)
    def IsPointInMatrix(self,p, board_border):
        """
        点是否范围内
        """
        p1 = board_border[0]
        p2 = board_border[1]
        p3 = board_border[2]
        p4 = board_border[3]
        isPointIn = self.GetCross(p1, p2, p) * self.GetCross(p3, p4, p) >= 0 and self.GetCross(p2, p3, p) * self.GetCross(p4, p1, p) >= 0
        return isPointIn

# def parse_opt():
#     '''
#     - `weights`: 模型权重文件的路径或 Triton URL。
#     - `source`: 图像或视频文件的路径或 URL，或者摄像头编号（0 表示默认摄像头）。
#     - `data`: 数据集配置文件的路径，其中包括类别名称。
#     - `imgsz`: 图像尺寸（高度，宽度），用于推断过程中的图像缩放。
#     - `conf`: 置信度阈值。
#     - `iou`: 非极大值抑制 (NMS) 的重叠阈值。
#     - `max_det`: 每张图像中最多的检测框数量。
#     - `device`: 使用的设备，可以是 cuda 设备（如 0 表示使用第一张显卡）或 cpu。
#     - `view_img`: 是否显示检测结果图像。
#     - `save_txt`: 是否将检测结果保存为 .txt 文件。
#     - `save_conf`: 是否保存检测结果中的置信度。
#     - `save_crop`: 是否保存检测结果中的裁剪框。
#     - `nosave`: 是否不保存图像或视频。
#     - `agnostic_nms`: 是否进行类别无关的 NMS。
#     - `augment`: 是否进行数据增强。
#     - `visualize`: 是否在推断过程中可视化特征。
#     - `project`: 结果保存的项目路径。
#     - `name`: 结果保存的项目名称。
#     - `exist_ok`: 如果项目路径已经存在，是否覆盖它。
#     - `line_thickness`: 绘制边界框的线条粗细。
#     - `hide_labels`: 是否隐藏标签。
#     - `hide_conf`: 是否隐藏置信度。
#     - `half`: 是否使用 FP16 的半精度推断。
#     - `dnn`: 是否使用 OpenCV DNN 进行 ONNX 推断。
#     - `vid_stride`: 视频帧率的步长。
#     - `retina_masks`: 是否使用高分辨率的掩码。
#     '''
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--weights', type=str, default=ROOT / 'best.pt',
#                         help='model path or triton URL')
#     parser.add_argument('--source', type=str, default=ROOT / '441.jpg',
#                         help='file/dir/URL/glob/screen/0(webcam)')
#     parser.add_argument('--view-img', default=True, action='store_true', help='View the prediction images')
#     parser.add_argument('--data', type=str, default=ROOT / 'data/618A.yaml', help='(optional) dataset.yaml path')
#     parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
#     parser.add_argument('--save-txt', default=True, action='store_true',
#                         help='Save the results in a txt file')
#     parser.add_argument('--conf', type=float, default=0.25, help='confidence threshold')
#     parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
#     parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
#     parser.add_argument('--save-conf', default=True, action='store_true', help='Save the condidence scores')
#     parser.add_argument('--save-crop', default=True, action='store_true', help='Save the crop')
#     parser.add_argument('--hide-labels', default=False, action='store_true', help='Hide the labels')
#     parser.add_argument('--hide-conf', default=False, action='store_true', help='Hide the confidence scores')
#     parser.add_argument('--vid-stride', default=False, action='store_true',
#                         help='Input video frame-rate stride')
#     parser.add_argument('--line-thickness', default=3, type=int, help='Bounding-box thickness (pixels)')
#     parser.add_argument('--visualize', default=False, action='store_true', help='Visualize model features')
#     parser.add_argument('--augment', default=False, action='store_true', help='Augmented inference')
#     parser.add_argument('--agnostic-nms', default=False, action='store_true', help='Class-agnostic NMS')
#     parser.add_argument('--retina-masks', default=False, action='store_true', help='High resolution masks')
#     parser.add_argument('--userId', default='VisionMaker', type=str, help='MES System account')
#     parser.add_argument('--gpuOpenOrClose', default=False)
#
#     opt = parser.parse_args()
#     opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
#     print_args(vars(opt))
#     return opt


# def main(opt):
#     yolo.run()

def CoordinateSystemConversion(Coordinate_data):
    Ox = float(Coordinate_data[0])
    Oy = float(Coordinate_data[1])
    Lx = float(Coordinate_data[2])
    Ly = float(Coordinate_data[3])
    if (Ox >= 0) and (Oy >= 0) and (Lx >= 0) and (Ly >= 0):
        D = math.sqrt((Lx - Ox) * (Lx - Ox) + (Ly - Oy) * (Ly - Oy))
        K = 70.0 / D
        sin_theta = (Lx - Ox) / D
        cos_theta = (Ly - Oy) / D
        # print(sin_theta, cos_theta)
        spin = np.asmatrix([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
        mov = np.asmatrix([[1, 0, -Ox], [0, 1, -Oy], [0, 0, 1]])
        getnewOxy = spin * mov
        return getnewOxy, K
    else:
        return False


def CoordinateSystemConversion_80(Coordinate_data):
    Ox = float(Coordinate_data[0])
    Oy = float(Coordinate_data[1])
    Lx = float(Coordinate_data[2])
    Ly = float(Coordinate_data[3])
    if (Ox >= 0) and (Oy >= 0) and (Lx >= 0) and (Ly >= 0):
        D = math.sqrt((Lx - Ox) * (Lx - Ox) + (Ly - Oy) * (Ly - Oy))
        K = 100.0 / D
        sin_theta = (Lx - Ox) / D
        cos_theta = (Ly - Oy) / D
        # print(sin_theta, cos_theta)
        spin = np.asmatrix([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
        mov = np.asmatrix([[1, 0, -Ox], [0, 1, -Oy], [0, 0, 1]])
        getnewOxy = spin * mov
        return getnewOxy, K
    else:
        return False


def CompareVectorDistance(Xvector, Yvector, Pvector, Qvector):
    XP = [Pvector[0] - Xvector[0], Pvector[1] - Xvector[1]]
    XQ = [Qvector[0] - Xvector[0], Qvector[1] - Xvector[1]]
    XY = [Yvector[0] - Xvector[0], Yvector[1] - Xvector[1]]
    distenceXP = abs(XP[0] * XY[0] + XP[1] * XY[1])
    distenceXQ = abs(XQ[0] * XY[0] + XQ[1] * XY[1])
    if (distenceXP < distenceXQ):
        return True
    else:
        return False


def AnalyzeCoordinate(strCoordinate):
    Coordinate_data = strCoordinate.split(',')
    if (len(Coordinate_data) == 4):
        Coordinate_reslut = CoordinateSystemConversion(Coordinate_data)
        if (Coordinate_reslut != False):
            NewOxy, K = Coordinate_reslut
            return NewOxy, K
        else:
            return False
    else:
        return False


def AnalyzeCoordinate_80(strCoordinate):
    Coordinate_data = strCoordinate.split(',')
    if (len(Coordinate_data) == 4):
        Coordinate_reslut = CoordinateSystemConversion_80(Coordinate_data)
        if (Coordinate_reslut != False):
            NewOxy, K = Coordinate_reslut
            return NewOxy, K
        else:
            return False
    else:
        return False

@app.get('/materials')
def MaterialsCatch():
    global newOxy
    global k
    global last_time
    # global MaterialsCount
    print( "收到请求识别颗粒指令:")
    # MaterialsCount += 1
    if newOxy.any() != None and k != None:
        # catchres = vadio.model1vadiocatch(newOxy, k, cameraOnRobot1)
        catchres = yolo.model1vadiocatch(newOxy, k)
        print(catchres)
        # catchres = '5'
        if catchres == None:
            return {'resflag': '3','res':'未识别'}
        last_time = datetime.datetime.now() # 更新上一次识别颗粒的时间
        return {'resflag': '1', 'res': catchres}
    else:
        return {'resflag': '-1','res':'5'}


@app.get('/agvpoint1')
def agvpoint1Catch():
    print("收到请求识别托盘指令:")
    global AGVOxy1
    global kagv1
    if AGVOxy1.any() != None and kagv1 != None:
        catchres = yolo.model2vadiocatch(AGVOxy1, kagv1)
        print(catchres)
        if catchres == None:
            return {'resflag': '3'}
        return {'resflag': '1', 'res': catchres}
    else:
        return {'resflag': '-1'}


@app.get('/agvpoint2')
def agvpoint2Catch():
    print("收到请求识别托盘指令:")
    global AGVOxy2
    global kagv2
    if AGVOxy2.any() != None and kagv2 != None:
        catchres = yolo.model2vadiocatch(AGVOxy2, kagv2)
        print(catchres)
        if catchres == None:
            return {'resflag': '3'}
        return {'resflag': '1', 'res': catchres}
    else:
        return {'resflag': '-1'}

if __name__ == "__main__":
    # main(opt)
    newOxy = k = None
    AGVOxy1 = kagv1 = None
    AGVOxy2 = kagv2 = None
    # opt = parse_opt()
    # opt = {
    #     "gpuOpenOrClose": 1,
    #     "runH5File": "best.pt",
    #     "identifyScore": 0.25,
    #     "identifyIou": 0.45,
    #     "cameraOnRobot1": 1,
    #     "cameraOnRobot2": 0,
    #     "CoordinateOfMaterials": "335.0,400.5,173.0,234.5",
    #     "CoordinateOfRobot1": "343.0,349.0,340.0,164.5",
    #     "CoordinateOfRobot2": "335.0,371.5,335.0,188.5",
    #     'board_border': '196,131,431,127,444,374,196,373', # 识别区域 p1:左上 p2:右上 p3:右下 p4:左下
    #     # board_border: [(196,131),(431,127),(444,374),(196,373)] # 识别区域 p1:左上 p2:右上 p3:右下 p4:左下
    #     "userId": "VisionMaker"
    # }
    opt = {
        "gpuOpenOrClose": 1 if (str(sys.argv[1]) == "True") else -1,
        "runH5File": str(sys.argv[2]),
        "identifyScore": float(sys.argv[3]),
        "identifyIou": float(sys.argv[4]),
        "cameraOnRobot1": 1 if (int(sys.argv[5]) == 1) else 0,
        "cameraOnRobot2": 1 if (int(sys.argv[6]) == 1) else 0,
        "CoordinateOfMaterials": str(sys.argv[7]),
        "CoordinateOfRobot1": str(sys.argv[8]),
        "CoordinateOfRobot2": str(sys.argv[9]),
        'board_border': '203,133,439,138,443,374,196,378',  # 识别区域 p1:左上 p2:右上 p3:右下 p4:左下
        "userId": str(sys.argv[10])
    }
    try:
        yolo = Yolo(opt)
        yolo.run()

        Materials_res = AnalyzeCoordinate_80(opt['CoordinateOfMaterials'])
        Robot1_res = AnalyzeCoordinate(opt['CoordinateOfRobot1'])
        Robot2_res = AnalyzeCoordinate(opt['CoordinateOfRobot2'])
        if (Materials_res != False):
            newOxy, k = Materials_res
        if (Robot1_res != False):
            AGVOxy1, kagv1 = Robot1_res
        if (Robot2_res != False):
            AGVOxy2, kagv2 = Robot2_res
        #
        if (Materials_res == False and Robot1_res == False and Robot2_res == False):
            print("Missing Data")

        # MaterialsCatch()
        # agvpoint1Catch()

        # 前端页面url
        origins = ["*"]

        # 后台api允许跨域
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        uvicorn.run(app=app,
                    host="0.0.0.0",
                    port=30079)
    finally:
        os.system('pause')
