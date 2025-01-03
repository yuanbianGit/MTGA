import cv2
import copy
import torch.nn.functional as F
import numpy as np
import collections.abc as container_abcs
from itertools import repeat
from models.vit_pytorch import vit_base_patch16_224_TransReID
from torchvision import  transforms
import torch
from torch import nn


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, bn_momentum=0.1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class StageModule(nn.Module):
    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()
        self.stage = stage
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
                BasicBlock(w, w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        # for each output_branches (i.e. each branch in all cases but the very last one)
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.stage):  # for each branch
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())  # Used in place of "None" because it is callable
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(1, 1), stride=(1, 1), bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Upsample(scale_factor=(2.0 ** (j - i)), mode='nearest'),
                    ))
                elif i > j:
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                      bias=False),
                            nn.BatchNorm2d(c * (2 ** j), eps=1e-05, momentum=0.1, affine=True,
                                           track_running_stats=True),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                  bias=False),
                        nn.BatchNorm2d(c * (2 ** i), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        assert len(self.branches) == len(x)

        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        for i in range(len(self.fuse_layers)):
            for j in range(0, len(self.branches)):
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused


class HRNet(nn.Module):
    def __init__(self, c=48, nof_joints=17, bn_momentum=0.1):
        super(HRNet, self).__init__()

        # Input (stem net)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1 (layer1)      - First group of bottleneck (resnet) modules
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # Fusion layer 1 (transition1)      - Creation of the first two branches (one full and one half resolution)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c, eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(256, c * (2 ** 1), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 1), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),
        ])

        # Stage 2 (stage2)      - Second module with 1 group of bottleneck (resnet) modules. This has 2 branches
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 2 (transition2)      - Creation of the third branch (1/4 resolution)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 1), c * (2 ** 2), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 2), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 3 (stage3)      - Third module with 4 groups of bottleneck (resnet) modules. This has 3 branches
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # Fusion layer 3 (transition3)      - Creation of the fourth branch (1/8 resolution)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * (2 ** 2), c * (2 ** 3), kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(c * (2 ** 3), eps=1e-05, momentum=bn_momentum, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
            )),  # ToDo Why the new branch derives from the "upper" branch only?
        ])

        # Stage 4 (stage4)      - Fourth module with 3 groups of bottleneck (resnet) modules. This has 4 branches
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # Final layer (final_layer)
        self.final_layer = nn.Conv2d(c, nof_joints, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list (# == nof branches)

        x = self.stage2(x)
        # x = [trans(x[-1]) for trans in self.transition2]    # New branch derives from the "upper" branch only
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x)
        # x = [trans(x) for trans in self.transition3]    # New branch derives from the "upper" branch only
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x)

        x = self.final_layer(x[0])

        return x


# if __name__ == '__main__':
#     # model = HRNet(48, 17, 0.1)
#     model = HRNet(32, 17, 0.1)

#     # print(model)

#     model.load_state_dict(
#         # torch.load('./weights/pose_hrnet_w48_384x288.pth')
#         torch.load('./weights/pose_hrnet_w32_256x192.pth')
#     )
#     print('ok!!')

#     if torch.cuda.is_available() and False:
#         torch.backends.cudnn.deterministic = True
#         device = torch.device('cuda:0')
#     else:
#         device = torch.device('cpu')

#     print(device)

#     model = model.to(device)

#     y = model(torch.ones(1, 3, 384, 288).to(device))
#     print(y.shape)
#     print(torch.min(y).item(), torch.mean(y).item(), torch.max(y).item())

class SimpleHRNet:
    """
    SimpleHRNet class.

    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(256, 128),
                 #384, 288
                 interpolation=cv2.INTER_CUBIC,
                 multiperson=False,
                 return_heatmaps=True,
                 return_bounding_boxes=False,
                 max_batch_size=32,
                 yolo_model_def="./models/detectors/yolo/config/yolov3.cfg",
                 yolo_class_path="./models/detectors/yolo/data/coco.names",
                 yolo_weights_path="./models/detectors/yolo/weights/yolov3.weights",
                 device=torch.device("cuda")):
        """
        Initializes a new SimpleHRNet object.
        HRNet (and YOLOv3) are initialized on the torch.device("device") and
        its (their) pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
            model_name (str): model name (HRNet or PoseResNet).
                Valid names for HRNet are: `HRNet`, `hrnet`
                Valid names for PoseResNet are: `PoseResNet`, `poseresnet`, `ResNet`, `resnet`
                Default: "HRNet"
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: (384, 288)
            interpolation (int): opencv interpolation algorithm.
                Default: cv2.INTER_CUBIC
            multiperson (bool): if True, multiperson detection will be enabled.
                This requires the use of a people detector (like YOLOv3).
                Default: True
            return_heatmaps (bool): if True, heatmaps will be returned along with poses by self.predict.
                Default: False
            return_bounding_boxes (bool): if True, bounding boxes will be returned along with poses by self.predict.
                Default: False
            max_batch_size (int): maximum batch size used in hrnet inference.
                Useless without multiperson=True.
                Default: 16
            yolo_model_def (str): path to yolo model definition file.
                Default: "./models/detectors/yolo/config/yolov3.cfg"
            yolo_class_path (str): path to yolo class definition file.
                Default: "./models/detectors/yolo/data/coco.names"
            yolo_weights_path (str): path to yolo pretrained weights file.
                Default: "./models/detectors/yolo/weights/yolov3.weights.cfg"
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.return_heatmaps = return_heatmaps
        self.return_bounding_boxes = return_bounding_boxes
        self.max_batch_size = max_batch_size
        self.yolo_model_def = yolo_model_def
        self.yolo_class_path = yolo_class_path
        self.yolo_weights_path = yolo_weights_path
        self.device = device

        if self.multiperson:
            from models.detectors.YOLOv3 import YOLOv3

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        # elif model_name in ('PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
        #     self.model = PoseResNet(resnet_size=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.model.eval()

        if not self.multiperson:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        else:
            self.detector = YOLOv3(model_def=yolo_model_def,
                                   class_path=yolo_class_path,
                                   weights_path=yolo_weights_path,
                                   classes=('person',),
                                   max_batch_size=self.max_batch_size,
                                   device=device)
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def predict(self, image):
        """
        Predicts the human pose on a single image or a stack of n images.

        Args:
            image (:class:`np.ndarray`):
                the image(s) on which the human pose will be estimated.

                image is expected to be in the opencv format.
                image can be:
                    - a single image with shape=(height, width, BGR color channel)
                    - a stack of n images with shape=(n, height, width, BGR color channel)

        Returns:
            :class:`np.ndarray` or list:
                a numpy array containing human joints for each (detected) person.

                Format:
                    if image is a single image:
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).
                    if image is a stack of n images:
                        list of n np.ndarrays with
                        shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (y position, x position, joint confidence).

                If self.return_heatmaps, the class returns a list with (heatmaps, human joints)
                If self.return_bounding_boxes, the class returns a list with (bounding boxes, human joints)
                If self.return_heatmaps and self.return_bounding_boxes, the class returns a list with
                    (heatmaps, bounding boxes, human joints)
        """
        if len(image.shape) == 3:
            return self._predict_single(image)
        elif len(image.shape) == 4:
            return self._predict_batch(image)
        else:
            raise ValueError('Wrong image format.')

    def _predict_single(self, image):
        if not self.multiperson:
            old_res = image.shape
            if self.resolution is not None:
                image = cv2.resize(
                    image,
                    (self.resolution[1], self.resolution[0]),  # (width, height)
                    interpolation=self.interpolation
                )

            images = self.transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(dim=0) # (h,w,c)
            # print('imgshape:',image.shape)
            boxes = np.asarray([[0, 0, old_res[1], old_res[0]]], dtype=np.float32)  # [x1, y1, x2, y2]
            heatmaps = np.zeros((1, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

        else:
            detections = self.detector.predict_single(image)

            nof_people = len(detections) if detections is not None else 0
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)
            heatmaps = np.zeros((nof_people, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

            if detections is not None:
                for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                    x1 = int(round(x1.item()))
                    x2 = int(round(x2.item()))
                    y1 = int(round(y1.item()))
                    y2 = int(round(y2.item()))

                    # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                    correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                    if correction_factor > 1:
                        # increase y side
                        center = y1 + (y2 - y1) // 2
                        length = int(round((y2 - y1) * correction_factor))
                        y1 = max(0, center - length // 2)
                        y2 = min(image.shape[0], center + length // 2)
                    elif correction_factor < 1:
                        # increase x side
                        center = x1 + (x2 - x1) // 2
                        length = int(round((x2 - x1) * 1 / correction_factor))
                        x1 = max(0, center - length // 2)
                        x2 = min(image.shape[1], center + length // 2)

                    boxes[i] = [x1, y1, x2, y2]
                    images[i] = self.transform(image[y1:y2, x1:x2, ::-1])

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: y, x, confidence
            for i, human in enumerate(out):
                heatmaps[i] = human
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]

    def _predict_batch(self, images):
        if not self.multiperson:
            old_res = images[0].shape
            # print('imge',imges.shape)

            # if self.resolution is not None:
            #     images_tensor = torch.empty(images.shape[0], 3, self.resolution[0], self.resolution[1])
            # else:
            #     images_tensor = torch.empty(images.shape[0], 3, images.shape[1], images.shape[2])

            # for i, image in enumerate(images):
            #     if self.resolution is not None:
            #         image = cv2.resize(
            #             image,
            #             (self.resolution[1], self.resolution[0]),  # (width, height)
            #             interpolation=self.interpolation
            #         )

            #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #     images_tensor[i] = self.transform(image)

            # images = images_tensor

            images = images # input tensor
            boxes = np.repeat(
                np.asarray([[0, 0, old_res[2], old_res[1]]], dtype=np.float32), len(images), axis=0
            )  # [x1, y1, x2, y2]
            heatmaps = np.zeros((len(images), self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

        else:
            image_detections = self.detector.predict(images)

            base_index = 0
            nof_people = int(np.sum([len(d) for d in image_detections if d is not None]))
            boxes = np.empty((nof_people, 4), dtype=np.int32)
            images_tensor = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)
            heatmaps = np.zeros((nof_people, self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                                dtype=np.float32)

            for d, detections in enumerate(image_detections):
                image = images[d]
                if detections is not None and len(detections) > 0:
                    for i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                        x1 = int(round(x1.item()))
                        x2 = int(round(x2.item()))
                        y1 = int(round(y1.item()))
                        y2 = int(round(y2.item()))

                        # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
                        correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
                        if correction_factor > 1:
                            # increase y side
                            center = y1 + (y2 - y1) // 2
                            length = int(round((y2 - y1) * correction_factor))
                            y1 = max(0, center - length // 2)
                            y2 = min(image.shape[0], center + length // 2)
                        elif correction_factor < 1:
                            # increase x side
                            center = x1 + (x2 - x1) // 2
                            length = int(round((x2 - x1) * 1 / correction_factor))
                            x1 = max(0, center - length // 2)
                            x2 = min(image.shape[1], center + length // 2)

                        boxes[base_index + i] = [x1, y1, x2, y2]
                        images_tensor[base_index + i] = self.transform(image[y1:y2, x1:x2, ::-1])

                    base_index += len(detections)

            images = images_tensor

        images = images.to(self.device)

        if images.shape[0] > 0:
            with torch.no_grad():
                if len(images) <= self.max_batch_size:
                    out = self.model(images)

                else:
                    out = torch.empty(
                        (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4),
                        device=self.device
                    )
                    for i in range(0, len(images), self.max_batch_size):
                        out[i:i + self.max_batch_size] = self.model(images[i:i + self.max_batch_size])
            # print('out.shape:',out.shape) #[bs, 17, H, W]
            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: y, x, confidence
            for i, human in enumerate(out):
                heatmaps[i] = human
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4)) #返回坐标位置(y, x)
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

            if self.multiperson:
                # re-add the removed batch axis (n)
                if self.return_heatmaps:
                    heatmaps_batch = []
                if self.return_bounding_boxes:
                    boxes_batch = []
                pts_batch = []
                index = 0
                for detections in image_detections:
                    if detections is not None:
                        pts_batch.append(pts[index:index + len(detections)])
                        if self.return_heatmaps:
                            heatmaps_batch.append(heatmaps[index:index + len(detections)])
                        if self.return_bounding_boxes:
                            boxes_batch.append(boxes[index:index + len(detections)])
                        index += len(detections)
                    else:
                        pts_batch.append(np.zeros((0, self.nof_joints, 3), dtype=np.float32))
                        if self.return_heatmaps:
                            heatmaps_batch.append(np.zeros((0, self.nof_joints, self.resolution[0] // 4,
                                                            self.resolution[1] // 4), dtype=np.float32))
                        if self.return_bounding_boxes:
                            boxes_batch.append(np.zeros((0, 4), dtype=np.float32))
                if self.return_heatmaps:
                    heatmaps = heatmaps_batch
                if self.return_bounding_boxes:
                    boxes = boxes_batch
                pts = pts_batch

            else:
                pts = np.expand_dims(pts, axis=1)

        else:
            boxes = np.asarray([], dtype=np.int32)
            if self.multiperson:
                pts = []
                for _ in range(len(image_detections)):
                    pts.append(np.zeros((0, self.nof_joints, 3), dtype=np.float32))
            else:
                raise ValueError  # should never happen

        res = list()
        if self.return_heatmaps:
            res.append(heatmaps)
        if self.return_bounding_boxes:
            res.append(boxes)
        res.append(pts)

        if len(res) > 1:
            return res
        else:
            return res[0]



def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    # dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def cosine_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    x_norm = torch.pow(x, 2).sum(1, keepdim=True).sqrt().expand(m, n)
    y_norm = torch.pow(y, 2).sum(1, keepdim=True).sqrt().expand(n, m).t()
    xy_intersection = torch.mm(x, y.t())
    dist = xy_intersection/(x_norm * y_norm)
    dist = (1. - dist) / 2
    return dist


def hard_example_mining(dist_mat, labels, return_inds=False):
    """For each anchor, find the hardest positive and negative sample.
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      return_inds: whether to return the indices. Save time if `False`(?)
    Returns:
      dist_ap: pytorch Variable, distance(anchor, positive); shape [N]
      dist_an: pytorch Variable, distance(anchor, negative); shape [N]
      p_inds: pytorch LongTensor, with shape [N];
        indices of selected hard positive samples; 0 <= p_inds[i] <= N - 1
      n_inds: pytorch LongTensor, with shape [N];
        indices of selected hard negative samples; 0 <= n_inds[i] <= N - 1
    NOTE: Only consider the case in which all labels have same num of samples,
      thus we can cope with all anchors in parallel.
    """

    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)

    # shape [N, N]
    is_pos = labels.expand(N, N).eq(labels.expand(N, N).t())
    is_neg = labels.expand(N, N).ne(labels.expand(N, N).t())

    # `dist_ap` means distance(anchor, positive)
    # both `dist_ap` and `relative_p_inds` with shape [N, 1]
    dist_ap, relative_p_inds = torch.max(
        dist_mat[is_pos].contiguous().view(N, -1), 1, keepdim=True)
    # print(dist_mat[is_pos].shape)
    # `dist_an` means distance(anchor, negative)
    # both `dist_an` and `relative_n_inds` with shape [N, 1]
    dist_an, relative_n_inds = torch.min(
        dist_mat[is_neg].contiguous().view(N, -1), 1, keepdim=True)
    # shape [N]
    dist_ap = dist_ap.squeeze(1)
    dist_an = dist_an.squeeze(1)

    if return_inds:
        # shape [N, N]
        ind = (labels.new().resize_as_(labels)
               .copy_(torch.arange(0, N).long())
               .unsqueeze(0).expand(N, N))
        # shape [N, 1]
        p_inds = torch.gather(
            ind[is_pos].contiguous().view(N, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(
            ind[is_neg].contiguous().view(N, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)
        return dist_ap, dist_an, p_inds, n_inds

    return dist_ap, dist_an


class TripletLoss(object):
    """
    Triplet loss using HARDER example mining,
    modified based on original triplet loss using hard example mining
    """

    def __init__(self, margin=None, hard_factor=0.0):
        self.margin = margin
        self.hard_factor = hard_factor
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def __call__(self, global_feat, labels, normalize_feature=False):
        if normalize_feature:
            global_feat = normalize(global_feat, axis=-1)
        dist_mat = euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = hard_example_mining(dist_mat, labels)

        dist_ap *= (1.0 + self.hard_factor)
        dist_an *= (1.0 - self.hard_factor)

        y = dist_an.new().resize_as_(dist_an).fill_(1)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = self.ranking_loss(dist_an - dist_ap, y)
        return loss, dist_ap, dist_an




class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + F.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm1 = self.norm1.requires_grad_()
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm2 = self.norm2.requires_grad_()
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, prototype, global_feat,
                     pos, query_pos):
        q = k = self.with_pos_embed(prototype, query_pos)
        prototype_2 = self.self_attn(q, k, value=prototype)[0]
        prototype = prototype + self.dropout1(prototype_2)
        prototype = self.norm1(prototype)
        out_prototype = self.multihead_attn(query=self.with_pos_embed(prototype, query_pos),
                                            key=self.with_pos_embed(global_feat, pos),
                                            value=global_feat)[0]
        prototype = prototype + self.dropout2(out_prototype)
        prototype = self.norm2(prototype)
        prototype = self.linear2(self.dropout(self.activation(self.linear1(prototype))))
        prototype = prototype + self.dropout3(prototype)
        prototype = self.norm3(prototype)
        return prototype

    def forward(self, prototype, global_feat, pos=None, query_pos=None):
        return self.forward_post(prototype, global_feat, pos, query_pos)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):  # nn.GELU
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, prototype, global_feat,
                pos=None, query_pos=None):
        output = prototype
        intermediate = []
        for layer in self.layers:
            output = layer(output, global_feat,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output


class PFD(nn.Module):

    def __init__(self, num_classes=702, camera_num=8, view_num=1):
        super(PFD, self).__init__()
        # model_path = cfg.MODEL.PRETRAIN_PATH
        # pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.in_planes = 768
        self.pose_dim = 2048

        # pose config
        self.pose = SimpleHRNet(48,
                                17,
                                '/data4/by/reid/github/MissRank_old/models/pfd/pose_hrnet_w48_384x288.pth',
                                model_name='HRNet',
                                resolution=(256, 128),
                                interpolation=cv2.INTER_CUBIC,
                                multiperson=False,
                                return_heatmaps=True,
                                return_bounding_boxes=False,
                                max_batch_size=32,
                                device=torch.device("cuda")
                                )

        # self.alphapose = SingleImageAlphaPose(alphapose_args, alphapose_cfg, device=torch.device("cuda"))

        self.skeleton_threshold = 0.3

        # decoderlayer config
        self.num_head = 8
        self.dim_forward = 2048
        self.decoder_drop =0.1
        self.drop_first = False

        # decoder config
        self.decoder_numlayer = 6
        self.decoder_norm = nn.LayerNorm(self.in_planes)

        # query setting
        self.num_query = 17
        self.query_embed = nn.Embedding(17, self.in_planes).weight

        # part view based decoder
        self.transformerdecoderlayer = TransformerDecoderLayer(self.in_planes, self.num_head, self.dim_forward,
                                                               self.decoder_drop, "relu", self.drop_first)
        self.transformerdecoder = TransformerDecoder(self.transformerdecoderlayer, self.decoder_numlayer,
                                                     self.decoder_norm)

        # print('using Transformer_type: {} as a encoder'.format(cfg.MODEL.TRANSFORMER_TYPE))

        # visual context encoder 
        # Thanks the authors of TransReID https://github.com/heshuting555/TransReID.git 
        self.base_vit = vit_base_patch16_224_TransReID(img_size=[256, 128], sie_xishu=3.0,
                                                            local_feature=True, camera=camera_num,
                                                            view=view_num, stride_size=[16, 16],
                                                            drop_path_rate=0.1)

        # print('Loading pretrained ImageNet model......from {}'.format(model_path))
        # if pretrain_choice == 'imagenet':
        #     self.base_vit.load_param(model_path)
        #     print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base_vit.blocks[-1]
        layer_norm = self.base_vit.norm
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_decoder = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_decoder.bias.requires_grad_(False)
        self.bottleneck_decoder.apply(weights_init_kaiming)
        self.non_skt_decoder = nn.BatchNorm1d(self.in_planes)
        self.non_skt_decoder.bias.requires_grad_(False)
        self.non_skt_decoder.apply(weights_init_kaiming)


        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

        for i in range(17):
            exec('self.classifier_{} = nn.Linear(self.in_planes, self.num_classes, bias=False)'.format(i + 1))
            exec('self.classifier_{}.apply(weights_init_classifier)'.format(i + 1))

        for i in range(17):
            exec('self.bottleneck_{} = nn.BatchNorm1d(self.in_planes)'.format(i + 1))
            exec('self.bottleneck_{}.bias.requires_grad_(False)'.format(i + 1))
            exec('self.bottleneck_{}.apply(weights_init_kaiming)'.format(i + 1))

        self.classifier_encoder = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_encoder.apply(weights_init_classifier)
        self.classifier_decoder = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_decoder.apply(weights_init_classifier)
        self.pose_decoder_linear = nn.Linear(self.pose_dim, self.in_planes)
        self.pose_avg = nn.AdaptiveAvgPool2d((1, self.in_planes))
        self.non_parts = nn.AdaptiveAvgPool2d((1, self.in_planes))
        self.decoder_global = nn.AdaptiveAvgPool2d((1, self.in_planes))

        for i in range(self.num_query):
            exec('self.classifier_decoder_{} = nn.Linear(self.in_planes, self.num_classes, bias=False)'.format(i + 1))
            exec('self.classifier_decoder_{}.apply(weights_init_classifier)'.format(i + 1))

        for i in range(self.num_query):
            exec('self.bottleneck_decoder_{} = nn.BatchNorm1d(self.in_planes)'.format(i + 1))
            exec('self.bottleneck_decoder_{}.bias.requires_grad_(False)'.format(i + 1))
            exec('self.bottleneck_decoder_{}.apply(weights_init_kaiming)'.format(i + 1))

    def forward(self, x, is_training=True):  # ht optinal

        bs, c, h, w = x.shape  # [batch, 3, 256, 128]

        # HRNet:
        heatmaps, joints = self.pose.predict(x)
        heatmaps = torch.from_numpy(heatmaps).cuda()  # [bs, 17, 64, 32]

        heatmaps = heatmaps.view(bs, heatmaps.shape[1], -1)  # [bs, 17, 2048]

        ttt = heatmaps.cpu().numpy()
        skt_ft = np.zeros((heatmaps.shape[0], heatmaps.shape[1]), dtype=np.float32)

        for i, heatmap in enumerate(ttt):  # [64]
            for j, joint in enumerate(heatmap):  # [17]

                if max(joint) < self.skeleton_threshold:
                    skt_ft[i][j] = 1  # Eq 4 in paper

        skt_ft = torch.from_numpy(skt_ft).cuda()  # [64, 17]

        pose_align_wt = self.pose_decoder_linear(heatmaps)  # [bs, 17, 768] FC

        heat_wt = self.pose_avg(heatmaps)  # [bs, 1, 768]

        features = self.base_vit(x, cam_label=0, view_label=0)  # [64, 129, 768] ViT

        # Input of decoder 
        decoder_value = features * heat_wt
        decoder_value = decoder_value.permute(1, 0, 2)

        # strip 
        feature_length = features.size(1) - 1  # 128
        patch_length = feature_length // self.num_query  # 128 // 17
        token = features[:, 0:1]
        x = features[:, 1:]

        sim_feat = []
        # Encoder group features
        for i in range(16):
            exec('b{}_local = x[:, patch_length*{}:patch_length*{}]'.format(i + 1, i, i + 1))

            exec('b{}_local_feat = self.b2(torch.cat((token, b{}_local), dim=1))'.format(i + 1, i + 1))
            # exec('print(b{}_local_feat.shape)'.format(i+1))
            exec('local_feat_{} = b{}_local_feat[:, 0]'.format(i + 1, i + 1))

            exec('sim_feat.append(local_feat_{})'.format(i + 1))

        b17_local = x[:, patch_length * 16:]
        b17_local_feat = self.b2(torch.cat((token, b17_local), dim=1))
        local_feat_17 = b17_local_feat[:, 0]
        sim_feat.append(local_feat_17)

        # inference list
        inf_encoder = []
        # BN
        for i in range(17):
            exec('local_feat_{}_bn = self.bottleneck_{}(local_feat_{})'.format(i + 1, i + 1, i + 1))
            exec('inf_encoder.append(local_feat_{}_bn/17)'.format(i + 1))

        feat = features[:, 0].unsqueeze(1) * heat_wt + features[:, 0].unsqueeze(1)

        feat = feat.squeeze(1)

        # f_gb feature from encoder
        global_out_feat = self.bottleneck(feat)  # [bs, 768]

        # part views 这是可学习的向量！
        query_embed = self.query_embed

        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)

        prototype = torch.zeros_like(query_embed)

        # part-view based decoder
        out = self.transformerdecoder(prototype, decoder_value, query_pos=query_embed)

        # part view features
        last_out = out.permute(1, 0, 2)  # [bs, num_query, 768]

        sim_decoder = torch.stack(sim_feat, dim=1)  # [bs, 17, 768]

        #  PFA 
        sim_decoder = PFA(sim_decoder, pose_align_wt)  # [bs 17 768]

        #  PVM
        decoder_feature, ind = PVM(sim_decoder, last_out)  # [bs num_query 768]

        decoder_gb = self.decoder_global(decoder_feature).squeeze(1)  # [bs, 1, 768]

        # non skt parts 
        out_non_parts = []
        # skt parts 
        out_skt_parts = []

        decoder_skt_feature = []

        decoder_non_feature = []

        for i in range(bs):
            non_skt_feat_list = []
            per_skt_feat_list = []

            skt_feat = skt_ft[i]  # [17]
            # non_zero_skt = torch.nonzero(skt_feat).squeeze(1) #[num]

            skt_part = skt_feat.cpu().numpy()
            skt_ind = np.argwhere(skt_part == 0).squeeze(1)  # [17-num] numpy type

            for j in range(decoder_feature.shape[1]):

                # version 1 use original heatmap label
                # if skt_feat[skt_ind[i][j]] == 0: 
                #     non_feat = decoder_feature[i, j, :]
                #     non_skt_feat_list.append(non_feat)

                if skt_feat[ind[i][j]] == 1:  # version 2 use PVM label
                    non_feat = decoder_feature[i, j, :]
                    non_skt_feat_list.append(non_feat)

                else:
                    skt_based_feat = decoder_feature[i, j, :]  # [768]
                    per_skt_feat_list.append(skt_based_feat)

            if len(non_skt_feat_list) == 0:
                zero_feature = torch.zeros_like(decoder_gb[i])
                non_skt_feat_list.append(zero_feature)  # TODO:
            non_skt_single = torch.stack(non_skt_feat_list, dim=0).unsqueeze(0)  # [1, len(nonzero), 768]、

            decoder_non_feature.append(non_skt_single)
            non_skt_single = self.non_parts(non_skt_single)  # [1, 1, 768]
            out_non_parts.append(non_skt_single)  # [[1,1,768], [1,1,768], ....] bs length

            if len(per_skt_feat_list) == 0:
                per_skt_feat_list.append(decoder_gb[i])  # TODO:
            skt_single = torch.stack(per_skt_feat_list, dim=0).unsqueeze(0)  # [1, x, 768]

            decoder_skt_feature.append(skt_single)
            skt_single = self.non_parts(skt_single)  # [1, 1, 768]
            out_skt_parts.append(skt_single)  # [[1,1,768], [1,1,768], ....] bs length

        last_non_parts = torch.cat(out_non_parts, dim=0)  # [bs, 1, 768]

        last_skt_parts = torch.cat(out_skt_parts, dim=0)  # [bs, 1, 768]

        # output high-confidence keypoint features
        decoder_out = self.bottleneck_decoder(last_skt_parts[:, 0])  # [bs, 768]

        # output non-skt-parts
        non_skt_parts = self.non_skt_decoder(last_non_parts[:, 0])

        # TODO:use last out or decoder out ?? 
        out_score = self.classifier_decoder(decoder_out)

        # Only high-confidence guided features are used to compute loss
        decoder_list = []

        # pad zeros for high-confidence guided features to self.num_query
        for i in decoder_skt_feature:
            if i.shape[1] < self.num_query:
                pad = torch.zeros((1, self.num_query - i.shape[1], self.in_planes)).to(i.device)
                pad_feat = torch.cat([i, pad], dim=1)  # [1, num_query, 768]
                decoder_list.append(pad_feat)
            else:
                decoder_list.append(i)

        decoder_lt = torch.cat(decoder_list, dim=0)  # [64, self.num_query, 768]

        decoder_feature = decoder_lt

        # decoder parts features
        decoder_feat = [decoder_out]
        decoder_inf = []
        for i in range(self.num_query):
            exec('b{}_deocder_local_feat = decoder_feature[:, {}]'.format(i + 1, i))
            exec('decoder_feat.append(b{}_deocder_local_feat)'.format(i + 1))
            exec('decoder_inf.append(b{}_deocder_local_feat/self.num_query)'.format(i + 1))

        # decoder BN
        for i in range(self.num_query):
            exec('decoder_local_feat_{}_bn = self.bottleneck_decoder_{}(b{}_deocder_local_feat)'.format(i + 1, i + 1,
                                                                                                        i + 1))

        encoder_feat = [global_out_feat] + sim_feat

        # if is_training:
        #     # encoder parts
        #     cls_score = self.classifier_encoder(global_out_feat)
        #
        #     encoder_score = [cls_score]
        #
        #     for i in range(17):
        #         exec('cls_score_{} = self.classifier_{}(local_feat_{}_bn)'.format(i + 1, i + 1, i + 1))
        #         exec('encoder_score.append(cls_score_{})'.format(i + 1))
        #
        #     decoder_score = [out_score]
        #
        #     # decoder parts
        #     for i in range(self.num_query):
        #         exec('decoder_cls_score_{} = self.classifier_decoder_{}(decoder_local_feat_{}_bn)'.format(i + 1, i + 1,
        #                                                                                                   i + 1))
        #         exec('decoder_score.append(decoder_cls_score_{})'.format(i + 1))
        #
        #     return encoder_score, encoder_feat, decoder_score, decoder_feat, non_skt_parts
        #
        # else:
            # Inferece concat
        inf_feat = [global_out_feat] + inf_encoder + [decoder_out] + decoder_inf
        inf_features = torch.cat(inf_feat, dim=1)

        return [None, inf_features]

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            # if 'w1' in i.replace('module.', '') or 'w2' in i.replace('module.', ''):
            #     continue
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def PVM(matrix, matrix1):
    '''
    @matrix shape [bs, 17, 768]
    @matrix1 shape [bs, x, 768] 
    '''

    assert matrix.shape[0] == matrix1.shape[0], 'Wrong shape'
    assert matrix.shape[2] == matrix1.shape[2], 'Wrong dimension'

    batch_size = matrix.shape[0]  # [bs, 17, 768]
    # skt_num = matrix.shape[1]
    final_sim = F.cosine_similarity(matrix.unsqueeze(2), matrix1.unsqueeze(1), dim=3)  # [bs, 17, x]

    _, ind = torch.max(final_sim, dim=2)  # ind.shape [bs, x]

    sim_match = []
    for i in range(batch_size):
        org_mat = matrix[i]  # [17, C]
        sim_mat = matrix1[i]  # [x, C]
        shuffle_mat = []

        for j in range(ind.shape[1]):
            new = org_mat[ind[i][j]] + sim_mat[j]  # [C]
            new = new.unsqueeze(0)
            shuffle_mat.append(new)

        bs_mat = torch.cat(shuffle_mat, dim=0)

        sim_match.append(bs_mat)

    final_feature = torch.stack(sim_match, dim=0)  # [bs, x, 768]?

    return final_feature, ind


def PFA(matrix, matrix1):
    '''
    @matrix shape [bs, 17, 768]
    @matrix1 shape [bs, 17, 768]

    '''
    assert matrix.shape[0] == matrix1.shape[0], 'Wrong shape'
    assert matrix.shape[1] == matrix1.shape[1], 'Wrong skt num'

    batch_size = matrix.shape[0]  # [bs, 17, 768]

    # skt_num = matrix.shape[1]

    pose_weighted_feat = matrix * matrix1  # [bs, 17, 768]

    final_sim = F.cosine_similarity(matrix.unsqueeze(2), pose_weighted_feat.unsqueeze(1), dim=3)  # [bs, 17, x]

    _, ind = torch.max(final_sim, dim=2)

    sim_match = []
    for i in range(batch_size):
        org_mat = matrix[i]  # [17, C]
        sim_mat = pose_weighted_feat[i]  # [17, C]
        shuffle_mat = []

        for j in range(ind.shape[1]):
            new = org_mat[j] + sim_mat[ind[i][j]]  # [C]
            new = new.unsqueeze(0)
            shuffle_mat.append(new)

        bs_mat = torch.cat(shuffle_mat, dim=0)

        sim_match.append(bs_mat)

    alignment_feat = torch.stack(sim_match, dim=0)  # [bs, 17, 768]?

    return alignment_feat




