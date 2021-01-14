import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from scipy import linalg
from scipy.spatial.transform import Rotation as R

from glob import glob
import cv2
import re
import os
import imgui
from imgui.integrations.glfw import GlfwRenderer
from pathlib import Path

from ctypes import c_wchar_p, windll  
from ctypes.wintypes import DWORD

AddDllDirectory = windll.kernel32.AddDllDirectory
AddDllDirectory.restype = DWORD
AddDllDirectory.argtypes = [c_wchar_p]
AddDllDirectory(r"C:\Program Files\Azure Kinect SDK v1.4.1\sdk\windows-desktop\amd64\release\bin") # modify path there if required

import pyk4a as k4a
from pyk4a import PyK4APlayback
from pyk4a import ImageFormat
import json

from helpers import colorize, convert_to_bgra_if_required

import torch
import torchvision
import utils

from PIL import Image, ImageOps
from torchvision.transforms import transforms as transforms

import time

def createBuffer(buffer, bufferType, size, usage):
    if buffer == -1:
        bufName = glGenBuffers(1)
    else:
        glDeleteBuffers(1, buffer)
        bufName = buffer
        bufName = glGenBuffers(1)

    glBindBuffer(bufferType, bufName)
    glBufferData(bufferType, size, None, usage)
    glBindBuffer(bufferType, 0)

    return bufName

def generateBuffers(bufferDict, depthWidth, depthHeight):
    
    p2pRedBufSize = depthWidth * depthHeight * 8 * 4 # 8 float32 per depth pixel for reduction struct
    p2pRedOutBufSize = depthWidth * depthHeight * 8 * 4 # 8 float32 per depth pixel for reduction struct

    bufferDict['p2pReduction'] = createBuffer(bufferDict['p2pReduction'], GL_SHADER_STORAGE_BUFFER, p2pRedBufSize, GL_DYNAMIC_DRAW)
    bufferDict['p2pRedOut'] = createBuffer(bufferDict['p2pRedOut'], GL_SHADER_STORAGE_BUFFER, p2pRedOutBufSize, GL_DYNAMIC_DRAW)

    return bufferDict


def createTexture(texture, target, internalFormat, levels, width, height, depth, minFilter, magFilter):

    if texture == -1:
        texName = glGenTextures(1)
    else:
        glDeleteTextures(int(texture))
        texName = texture
        texName = glGenTextures(1)

    glBindTexture(target, texName)
    #texture wrapping params
    glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
    glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
    #texture filtering params
    glTexParameteri(target, GL_TEXTURE_MIN_FILTER, minFilter)
    glTexParameteri(target, GL_TEXTURE_MAG_FILTER, magFilter)
    if target == GL_TEXTURE_1D:
        glTexStorage1D(target, levels, internalFormat, width)
    elif target == GL_TEXTURE_2D:
        glTexStorage2D(target, levels, internalFormat, width, height)
    elif target == GL_TEXTURE_3D or depth > 1:
        glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER)
        glTexStorage3D(target, levels, internalFormat, width, height, depth)

    #glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
    #glPixelStorei(GL_UNPACK_ROW_LENGTH, int(width))

    return texName

def generateTextures(textureDict, colwidth, colheight, depwidth, depheight):


    #lastColor
    textureDict['rawColor'] = createTexture(textureDict['rawColor'], GL_TEXTURE_2D, GL_RGBA8, 1, int(colwidth), int(colheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

    textureDict['lastColor'] = createTexture(textureDict['lastColor'], GL_TEXTURE_2D, GL_RGBA8, 3, int(colwidth), int(colheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['nextColor'] = createTexture(textureDict['nextColor'], GL_TEXTURE_2D, GL_RGBA8, 3, int(colwidth), int(colheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    
    textureDict['rawDepth'] = createTexture(textureDict['rawDepth'], GL_TEXTURE_2D, GL_R16, 1, int(depwidth), int(depheight), 1, GL_NEAREST, GL_NEAREST)
    
    textureDict['filteredDepth'] = createTexture(textureDict['filteredDepth'], GL_TEXTURE_2D, GL_R32F, 1, int(depwidth), int(depheight), 1, GL_NEAREST, GL_NEAREST)
    
    textureDict['xyLUT'] = createTexture(textureDict['xyLUT'], GL_TEXTURE_2D, GL_RG32F, 1, int(depwidth), int(depheight), 1, GL_NEAREST, GL_NEAREST)

    textureDict['lastDepth'] = createTexture(textureDict['lastDepth'], GL_TEXTURE_2D, GL_R32F, 3, int(depwidth), int(depheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['nextDepth'] = createTexture(textureDict['nextDepth'], GL_TEXTURE_2D, GL_R32F, 3, int(depwidth), int(depheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    
    textureDict['mappingC2D'] = createTexture(textureDict['mappingC2D'], GL_TEXTURE_2D, GL_RG16, 3, int(colwidth), int(colheight), 1, GL_NEAREST, GL_NEAREST)
    textureDict['mappingD2C'] = createTexture(textureDict['mappingD2C'], GL_TEXTURE_2D, GL_RG16, 3, int(colwidth), int(colheight), 1, GL_NEAREST, GL_NEAREST)

    textureDict['refVertex'] = createTexture(textureDict['refVertex'], GL_TEXTURE_2D, GL_RGBA32F, 3, int(depwidth), int(depheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['virtualVertex'] = createTexture(textureDict['virtualVertex'], GL_TEXTURE_2D, GL_RGBA32F, 3, int(depwidth), int(depheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    
    textureDict['refNormal'] = createTexture(textureDict['refNormal'], GL_TEXTURE_2D, GL_RGBA32F, 3, int(depwidth), int(depheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['virtualNormal'] = createTexture(textureDict['virtualNormal'], GL_TEXTURE_2D, GL_RGBA32F, 3, int(depwidth), int(depheight), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

    textureDict['volume'] = createTexture(textureDict['volume'], GL_TEXTURE_3D, GL_RG16F, 1, 128, 128, 128, GL_NEAREST, GL_NEAREST)

    textureList['tracking'] = createTexture(textureList['tracking'], GL_TEXTURE_2D, GL_RGBA32F, 3, int(depwidth), int(depheight), 1, GL_NEAREST, GL_NEAREST)
    #nextFlowMap
    #textureList[5] = createTexture(textureList[5], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #sparseFlowMap
    #textureList[6] = createTexture(textureList[6], GL_TEXTURE_2D, GL_RGBA32F, maxLevels, int(width / 4), int(height / 4), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #densificationFlowMap
    #textureList[7] = createTexture(textureList[7], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

 
    

	# Allocate the immutable GPU memory storage -more efficient than mutable memory if you are not going to change image size after creation

    return textureDict

def createXYLUT(playback, textureDict, depthWidth, depthHeight):

    xyTable = np.zeros((depthHeight, depthWidth, 2), dtype = "float")

    for x in range(depthHeight):
        for y in range(depthWidth):
            point = float(x), float(y)
            converted = playback.calibration.convert_2d_to_3d(point, 1.0, k4a.CalibrationType.DEPTH, k4a.CalibrationType.DEPTH)
            if np.isnan(converted).any():
                xyTable[x, y] = -1, -1  
            else:
                xyTable[x, y] = converted[0], converted[1]  
    
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['xyLUT'])
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(depthWidth), int(depthHeight), GL_RG, GL_FLOAT, xyTable)


def bilateralFilter(shaderDict, textureDict, depthWidth, depthHeight):
    glUseProgram(shaderDict['bilateralFilterShader'])
    # set logic for using filtered or unfiltered TODO

    glBindImageTexture(0, textureDict['rawDepth'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R16UI) 
    glBindImageTexture(1, textureDict['lastDepth'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F) 
    glBindImageTexture(2, textureDict['filteredDepth'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F) 

    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "depthScale"), 0.001)
    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "sigma"), 10.0)
    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "bSigma"), 0.05)

    compWidth = int((depthWidth/32.0)+0.5)
    compHeight = int((depthHeight/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)    


def alignDepthColor(shaderDict, textureDict, colorWidth, colorHeight, depthWidth, depthHeight):
    glUseProgram(shaderDict['alignDepthColorShader'])

    glBindImageTexture(0, textureDict['filteredDepth'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F) 
    glBindImageTexture(1, textureDict['lastColor'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F) 
    glBindImageTexture(2, textureDict['refVertex'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    compWidth = int((width/32.0)+0.5)
    compHeight = int((height/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def depthToVertex(shaderDict, textureDict, width, height):

    glUseProgram(shaderDict['depthToVertexShader'])

    glBindImageTexture(0, textureDict['filteredDepth'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F) 
    glBindImageTexture(1, textureDict['xyLUT'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F) 
    glBindImageTexture(2, textureDict['refVertex'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    glUniform1f(glGetUniformLocation(shaderDict['depthToVertexShader'], "minDepth"), 0.001)
    glUniform1f(glGetUniformLocation(shaderDict['depthToVertexShader'], "maxDepth"), 10.0)
    glUniform2f(glGetUniformLocation(shaderDict['depthToVertexShader'], "bottomLeft"), 0, 0)
    glUniform2f(glGetUniformLocation(shaderDict['depthToVertexShader'], "topRight"), 640, 576)

    compWidth = int((width/32.0)+0.5)
    compHeight = int((height/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def vertexToNormal(shaderDict, textureDict, depthWidth, depthHeight):

    glUseProgram(shaderDict['vertexToNormalShader'])

    glBindImageTexture(0, textureDict['refVertex'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['refNormal'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    compWidth = int((depthWidth/32.0)+0.5)
    compHeight = int((depthHeight/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def p2pTrack(shaderDict, textureDict, bufferDict, currPose, level):
    glUseProgram(shaderDict['trackP2PShader'])

    glBindImageTexture(0, textureDict['refVertex'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['refNormal'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)

    glBindImageTexture(2, textureDict['virtualVertex'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(3, textureDict['virtualNormal'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 

    glBindImageTexture(4, textureDict['tracking'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['p2pReduction'])

    glUniform1i(glGetUniformLocation(shaderDict['trackP2PShader'], "mip"), fusionConfig['level'])

    glUniformMatrix4fv(glGetUniformLocation(shaderDict['trackP2PShader'], "K"), 1, False, K)
    glUniformMatrix4fv(glGetUniformLocation(shaderDict['trackP2PShader'], "invK"), 1, False, np.invert(K))

    glUniformMatrix4fv(glGetUniformLocation(shaderDict['trackP2PShader'], "T"), 1, False, currPose)
    glUniformMatrix4fv(glGetUniformLocation(shaderDict['trackP2PShader'], "invT"), 1, False, np.invert(currPose))

    compWidth = int((depthWidth/32.0)+0.5)
    compHeight = int((depthHeight/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

#def p2pReduce():

#def p2pGetReduction():        


def raycastVolume(shaderDict, textureDict, fusionConfig):
    glUseProgram(shaderDict['raycastVolumeShader'])

    glBindImageTexture(0, textureDict['volume'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG16F) 
    glBindImageTexture(1, textureDict['refVertex'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['refNormal'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "nearPlane"), fusionConfig['nearPlane'])
    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "farPlane"), fusionConfig['farPlane'])
    glUniform3f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "volDim"), fusionConfig['volDim'][0], fusionConfig['volDim'][1], fusionConfig['volDim'][2])
    glUniform3f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "volSize"), fusionConfig['volSize'][0], fusionConfig['volSize'][1], fusionConfig['volSize'][2])

    step = np.max(fusionConfig['volDim']) / np.max(fusionConfig['volSize'])
    largeStep = 0.375 # dont know why
    #dMin = -fusionConfig['volDim'][0] / 20.0
    #dMax = fusionConfig['volDim'][0] / 10.0
    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "step"), step)
    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "largeStep"), largeStep)

    compWidth = int((fusionConfig['volSize'][0]/32.0)+0.5) 
    compHeight = int((fusionConfig['volSize'][1]/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def integrateVolume(shaderDict, textureDict):
    glUseProgram(shaderDict['integrateShader'])

def runP2P(shaderDict, textureDict, bufferDict, fusionConfig, currPose, integrateFlag):
    pose = np.array((4, 4), dtype = 'float')
    raycastVolume(shaderDict, textureDict, fusionConfig)

    for level in range(np.size(fusionConfig['iters']), -1, -1):
        for iter in range(fusionConfig['iters'][level]):
            A = np.array((6, 6), dtype = 'double')
            b = np.array((6, 1), dtype = 'double')
            
            p2pTrack(shaderDict, textureDict, bufferDict, currPose, level)

            # p2pReduce(bufferDict)

            # A, b, AE, icpCount = p2pGetReduction(bufferDict)

            # result = np.array((6, 1), dtype = 'double')

            # result = linalg.solve(A, b)

            # deltaR = R.from_rotvec(result[3], result[4], result[5])
            # delta = np.array((4, 4), dtype = 'float')
            # delta[:3, :3] = deltaR.as_matrix()

            # delta[3, 0] = result[0]
            # delta[3, 1] = result[1]
            # delta[3, 2] = result[2]

            # pose = delta * pose

            # resNorm = linalg.norm(result)

            # if (resNorm < 1e-5 and resNorm != 0):
            #     break

    #if integrateFlag == True:
    #    integrateVolume(shaderDict, textureDict)


def render(VAO, window, shaderDict, textureDict):

    glUseProgram(shaderDict['renderShader'])
    glClear(GL_COLOR_BUFFER_BIT)

    w, h = glfw.get_framebuffer_size(window)
    xpos = 0
    ypos = 0
    xwidth = float(w) / 3.0


    #render depth
    opts = 1 << 0 | 0 << 1 | 0 << 2 | 0 << 3 | 0 << 4
    glViewport(int(xpos), int(ypos), int(xwidth),h)
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['filteredDepth'])
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "isYFlip"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderType"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderOptions"), opts)
    glUniform2f(glGetUniformLocation(shaderDict['renderShader'], "depthRange"), 0.0, 5.0)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    #render normals
    opts = 0 << 0 | 1 << 1 | 0 << 2 | 0 << 3 | 0 << 4
    xpos = w / 3.0
    glViewport(int(xpos), int(ypos), int(xwidth),h)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textureDict['refNormal'])
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "isYFlip"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderType"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderOptions"), opts)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)   

    #render color
    opts = 0 << 0 | 0 << 1 | 1 << 2 | 0 << 3 | 0 << 4
    xpos = 2 * w / 3.0
    glViewport(int(xpos), int(ypos), int(xwidth),h)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, textureDict['lastColor'])
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "isYFlip"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderType"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderOptions"), opts)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)   

def main():

    # # transform to convert the image to tensor
    # transform = transforms.Compose([
    #     transforms.ToTensor()
    # ])
    # # initialize the model
    # model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,
    #                                                             num_keypoints=17)
    # # set the computation device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # load the modle on to the computation device and set to eval mode
    # model.to(device).eval()

    # initialize glfw
    if not glfw.init():
        return
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    #creating the window
    window = glfw.create_window(1600, 900, "PyGLFusion", None, None)
    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    imgui.create_context()
    impl = GlfwRenderer(window)

    depthWidth = 640
    depthHeight = 576

    colorWidth = 1920
    colorHeight = 1080

  
    # rendering
    glClearColor(0.2, 0.3, 0.2, 1.0)

    #           positions        texture coords
    quad = [   -1.0, -1.0, 0.0,  0.0, 0.0,
                1.0, -1.0, 0.0,  1.0, 0.0,
                1.0,  1.0, 0.0,  1.0, 1.0,
               -1.0,  1.0, 0.0,  0.0, 1.0]

    quad = np.array(quad, dtype = np.float32)

    indices = [0, 1, 2,
               2, 3, 0]

    indices = np.array(indices, dtype= np.uint32)

    screenVertex_shader = (Path(__file__).parent / 'shaders/screenQuad.vert').read_text()

    screenFragment_shader = (Path(__file__).parent / 'shaders/screenQuad.frag').read_text()

    renderShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(screenVertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(screenFragment_shader, GL_FRAGMENT_SHADER))

    # set up VAO and VBO for full screen quad drawing calls
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)

    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 80, quad, GL_STATIC_DRAW)

    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 24, indices, GL_STATIC_DRAW)

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)




    # shaders

    bilateralFilter_shader = (Path(__file__).parent / 'shaders/bilateralFilter.comp').read_text()
    bilateralFilterShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(bilateralFilter_shader, GL_COMPUTE_SHADER))

    alignDepthColor_shader = (Path(__file__).parent / 'shaders/alignDepthColor.comp').read_text()
    alignDepthColorShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(alignDepthColor_shader, GL_COMPUTE_SHADER))

    depthToVertex_shader = (Path(__file__).parent / 'shaders/depthToVertex.comp').read_text()
    depthToVertexShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(depthToVertex_shader, GL_COMPUTE_SHADER))

    vertexToNormal_shader = (Path(__file__).parent / 'shaders/vertexToNormal.comp').read_text()
    vertexToNormalShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertexToNormal_shader, GL_COMPUTE_SHADER))

    raycast_shader = (Path(__file__).parent / 'shaders/raycast.comp').read_text()
    raycastShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(raycast_shader, GL_COMPUTE_SHADER))

    trackP2P_shader = (Path(__file__).parent / 'shaders/p2pTrack.comp').read_text()
    trackP2PShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(trackP2P_shader, GL_COMPUTE_SHADER))


    playback = PyK4APlayback("C:\data\outSess1.mkv")
    playback.open()

    cal = json.loads(playback.calibration_raw)
    depthCal = cal["CalibrationInformation"]["Cameras"][0]["Intrinsics"]["ModelParameters"]
    colorCal = cal["CalibrationInformation"]["Cameras"][1]["Intrinsics"]["ModelParameters"]
    # https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/61951daac782234f4f28322c0904ba1c4702d0ba/src/transformation/mode_specific_calibration.c
    # from microsfots way of doing things, you have to do the maths here, rememberedding to -0.5f from cx, cy at the end
    K = np.eye(4, dtype='float')
    K[0][0] = depthCal[2] * cal["CalibrationInformation"]["Cameras"][0]["SensorWidth"] # fx
    K[1][1] = depthCal[3] * cal["CalibrationInformation"]["Cameras"][0]["SensorHeight"] # fy
    K[2][0] = (depthCal[0] * cal["CalibrationInformation"]["Cameras"][0]["SensorHeight"]) - 192.0 - 0.5 # cx
    K[2][1] = (depthCal[1] * cal["CalibrationInformation"]["Cameras"][0]["SensorHeight"]) - 180.0 - 0.5 # cy

    #playback.configuration["color_format"] == ImageFormat.COLOR_MJPG
    if playback.configuration["depth_mode"] == k4a.DepthMode.NFOV_UNBINNED:
        print("hello")

    shaderDict = {
        'renderShader' : renderShader,
        'bilateralFilterShader' : bilateralFilterShader,
        'alignDepthColorShader' : alignDepthColorShader,
        'depthToVertexShader' : depthToVertexShader,
        'vertexToNormalShader' : vertexToNormalShader,
        'raycastVolumeShader' : raycastShader
    }

    bufferDict = {
        'p2pReduction' : -1,
        'p2pRedOut' : -1
    }

    textureDict = {
        'rawColor' : -1,
        'lastColor' : -1,
        'nextColor' : -1,
        'rawDepth' :  -1,
        'filteredDepth' : -1,
        'lastDepth' : -1,
        'nextDepth' : -1,
        'refVertex' : -1,
        'virtualVertex' : -1,
        'refNormal' : -1,
        'virtualNormal' : -1,
        'mappingC2D' : -1,
        'mappingD2C' : -1,
        'xyLUT' : -1, 
        'tracking' : -1,
        'volume' : -1
    }

    fusionConfig = {
        'volSize' : (128, 128, 128),
        'volDim' : (1.0, 1.0, 1.0),
        'iters' : (2, 5, 10),
        'maxWeight' : 100.0,
        'distThresh' : 0.05,
        'normThresh' : 0.9,
        'nearPlane' : 0.1,
        'farPlane' : 5.0
    }

    cameraConfig = {
        'depthWidth' : 640,
        'depthHeight' : 576,
        'colorWidth' : 1920,
        'colorHeight' : 1080,
        'K' : 1
    }

    textureDict = generateTextures(textureDict, colorWidth, colorHeight, depthWidth, depthHeight)
    bufferDict = generateBuffers(bufferDict, depthWidth, depthHeight)

    colorMat = np.zeros((colorHeight, colorWidth, 3), dtype = "uint8")
    useColorMat = False 
    integrateFlag = False
    currPose = np.eye(4, dtype='float')

    # LUTs
    createXYLUT(playback, textureDict, depthWidth, depthHeight)

    while not glfw.window_should_close(window):

        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()

        try:
            capture = playback.get_next_capture()
            if capture.color is not None:
                if playback.configuration["color_format"] == ImageFormat.COLOR_MJPG:
                    colorMat = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                    useColorMat = True

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, textureDict['lastColor'])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(colorWidth), int(colorHeight), GL_RGB, GL_UNSIGNED_BYTE, (capture.color, colorMat)[useColorMat] )

            if capture.depth is not None:
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, textureDict['rawDepth'])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(depthWidth), int(depthHeight), GL_RED, GL_UNSIGNED_SHORT, capture.depth)

        except EOFError:
            break

        # #smallMat = cv2.pyrDown(colorMat)
        # rotMat = cv2.flip(colorMat, 0)

        # pil_image = Image.fromarray(rotMat).convert('RGB')
        # image = transform(pil_image)
        # image = image.unsqueeze(0).to(device)
        # #start_time = time.time()
        # with torch.no_grad():
        #     outputs = model(image)
        # #end_time = time.time()
        # #print((end_time - start_time) * 1000.0)
        # output_image = utils.draw_keypoints(outputs, rotMat)
        # cv2.imshow('Face detection frame', output_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break





        bilateralFilter(shaderDict, textureDict, depthWidth, depthHeight)
        depthToVertex(shaderDict, textureDict, depthWidth, depthHeight)
        #alignDepthColor(alignDepthColorShader, textureDict, colorWidth, colorHeight, depthWidth, depthHeight)
        vertexToNormal(shaderDict, textureDict, depthWidth, depthHeight)

        runP2P(shaderDict, textureDict, bufferDict, fusionConfig, currPose, integrateFlag)





        render(VAO, window, shaderDict, textureDict)

        imgui.render()

        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)        

    glfw.terminate()
    


if __name__ == "__main__":
    main()    