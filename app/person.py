from OpenGL.GL import *
import OpenGL.GL.shaders

from pathlib import Path

import cv2
import numpy as np

import json
#import trt_pose.coco
import torch
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

def coco_category_to_topology(coco_category):
    """Gets topology tensor from a COCO category
    """
    skeleton = coco_category['skeleton']
    K = len(skeleton)
    topology = torch.zeros((K, 4)).int()
    for k in range(K):
        topology[k][0] = 2 * k
        topology[k][1] = 2 * k + 1
        topology[k][2] = skeleton[k][0] - 1
        topology[k][3] = skeleton[k][1] - 1
    return topology

def init():

    import torch2trt
    from torch2trt import TRTModule

    with open('./models/human_pose.json', 'r') as f:
        human_pose = json.load(f)
    
    global topology
    topology = coco_category_to_topology(human_pose)

    global WIDTH
    WIDTH = 256
    global HEIGHT
    HEIGHT = 256

    #data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

    OPTIMIZED_MODEL = Path('./models/densenet121_baseline_att_256x256_B_epoch_160_trt.pth')

    global model_trt
    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))

    print('loaded model')

    global mean
    mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
    global std
    std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
    global device
    device = torch.device('cuda')

    global parse_objects
    parse_objects = ParseObjects(topology)
    global draw_objects
    draw_objects = DrawObjects(topology)

def preprocess(image, mean, std):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def getPose(textureDict, cameraConfig, frame):
    imageSmall = cv2.resize(frame, (WIDTH, HEIGHT))
    data = preprocess(imageSmall, mean, std)
    cmap, paf = model_trt(data)
    cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
    counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
    draw_objects(frame, counts, objects, peaks)
    print(counts)
    cv2.imshow('fr', frame)
    cv2.waitKey(1)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['rawColor'])
    img_data = np.array(frame.data, np.uint8)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), GL_BGR, GL_UNSIGNED_BYTE, img_data)    