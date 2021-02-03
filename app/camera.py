import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import glm
import numpy as np
import os

if os.name == 'nt':
    from ctypes import c_wchar_p, windll  
    from ctypes.wintypes import DWORD
    AddDllDirectory = windll.kernel32.AddDllDirectory
    AddDllDirectory.restype = DWORD
    AddDllDirectory.argtypes = [c_wchar_p]
    AddDllDirectory(r"C:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin") # modify path there if required

import pyk4a
from pyk4a import PyK4APlayback
from pyk4a import ImageFormat
from pyk4a import Config, PyK4A
import json


def start(useLiveCamera):
    global k4a

    if useLiveCamera == False:
        k4a = PyK4APlayback("C:\data\outSess1.mkv")
        k4a.open()
    else: 
        k4a = PyK4A(
            Config(
                color_resolution=pyk4a.ColorResolution.RES_1080P,
                depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
                synchronized_images_only=True,
            )
        )
        k4a.start()

    
    cal = json.loads(k4a.calibration_raw)
    depthCal = cal["CalibrationInformation"]["Cameras"][0]["Intrinsics"]["ModelParameters"]
    colorCal = cal["CalibrationInformation"]["Cameras"][1]["Intrinsics"]["ModelParameters"]
    # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/61951daac782234f4f28322c0904ba1c4702d0ba/src/transformation/mode_specific_calibration.c
    # from microsfots way of doing things, you have to do the maths here, rememberedding to -0.5f from cx, cy at the end
    # this should be set from the depth mode type, as the offsets are different, see source code in link
    #K = np.eye(4, dtype='float32')
    K = glm.mat4(1.0)
    K[0, 0] = depthCal[2] * cal["CalibrationInformation"]["Cameras"][0]["SensorWidth"] # fx
    K[1, 1] = depthCal[3] * cal["CalibrationInformation"]["Cameras"][0]["SensorHeight"] # fy
    K[2, 0] = (depthCal[0] * cal["CalibrationInformation"]["Cameras"][0]["SensorWidth"]) - 192.0 - 0.5 # cx
    K[2, 1] = (depthCal[1] * cal["CalibrationInformation"]["Cameras"][0]["SensorHeight"]) - 180.0 - 0.5 # cy

    invK = glm.inverse(K)

    colK = glm.mat4(1.0)
    colK[0, 0] = colorCal[2] * 1920.0# fx
    colK[1, 1] = colorCal[3] * 1440.0 # fy # why 1440, since we are 1080p? check the link, the umbers are there, im sure they make sense ...
    colK[2, 0] = (colorCal[0] * 1920.0) - 0 - 0.5 # cx
    colK[2, 1] = (colorCal[1] * 1440.0) - 180.0 - 0.5 # cy

    d2c = glm.mat4(
        cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][0], cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][3], cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][6], 0,
        cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][1], cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][4], cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][7], 0,
        cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][2], cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][5], cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Rotation"][8], 0,
        cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Translation"][0], cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Translation"][1], cal["CalibrationInformation"]["Cameras"][1]["Rt"]["Translation"][2], 1
        )

    c2d = glm.inverse(d2c)

    return d2c, c2d, K, invK, colK

def getFrames(useLiveCamera):
    if useLiveCamera == False:
        capture = k4a.get_next_capture()
    else:    
        capture = k4a.get_capture()

    return capture    


def stop():
    k4a.stop()
