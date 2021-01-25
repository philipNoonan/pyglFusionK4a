import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import glm

import ctypes

import gc

import math

import numpy as np
from scipy import linalg

from glob import glob
import cv2
import re
import os
import imgui
from imgui.integrations.glfw import GlfwRenderer
from pathlib import Path

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

#import torch
#import torchvision
#import utils

#from PIL import Image, ImageOps
#from torchvision.transforms import transforms as transforms

import time
global float32_data
float32_data = (ctypes.c_float * 256)()



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

def generateBuffers(bufferDict, cameraConfig):
    
    p2pRedBufSize = cameraConfig['depthWidth'] * cameraConfig['depthHeight'] * 8 * 4 # 8 float32 per depth pixel for reduction struct
    p2pRedOutBufSize = 32 * 4 * 4 # 32 outs per local group, upto 4 local groups running on highest def layer
    p2vRedBufSize = cameraConfig['depthWidth'] * cameraConfig['depthHeight'] * 9 * 4 # 9 float32 per depth pixel for reduction struct
    p2vRedOutBufSize = 32 * 8 * 4

    bufferDict['p2pReduction'] = createBuffer(bufferDict['p2pReduction'], GL_SHADER_STORAGE_BUFFER, p2pRedBufSize, GL_DYNAMIC_DRAW)
    bufferDict['p2pRedOut'] = createBuffer(bufferDict['p2pRedOut'], GL_SHADER_STORAGE_BUFFER, p2pRedOutBufSize, GL_DYNAMIC_DRAW)
    
    bufferDict['p2vReduction'] = createBuffer(bufferDict['p2vReduction'], GL_SHADER_STORAGE_BUFFER, p2vRedBufSize, GL_DYNAMIC_DRAW)
    bufferDict['p2vRedOut'] = createBuffer(bufferDict['p2vRedOut'], GL_SHADER_STORAGE_BUFFER, p2vRedOutBufSize, GL_DYNAMIC_DRAW)

    bufferDict['test'] = createBuffer(bufferDict['test'], GL_SHADER_STORAGE_BUFFER, 32, GL_DYNAMIC_DRAW)
    bufferDict['outBuf'] = createBuffer(bufferDict['outBuf'], GL_SHADER_STORAGE_BUFFER, 36 * 4, GL_DYNAMIC_DRAW)
    bufferDict['poseBuffer'] = createBuffer(bufferDict['poseBuffer'], GL_SHADER_STORAGE_BUFFER, 16 * 4, GL_DYNAMIC_DRAW)

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

def generateTextures(textureDict, cameraConfig, fusionConfig):

    numLevels = np.size(fusionConfig['iters'])
    #lastColor
    textureDict['rawColor'] = createTexture(textureDict['rawColor'], GL_TEXTURE_2D, GL_RGBA8, 1, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

    textureDict['lastColor'] = createTexture(textureDict['lastColor'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['nextColor'] = createTexture(textureDict['nextColor'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    
    textureDict['rawDepth'] = createTexture(textureDict['rawDepth'], GL_TEXTURE_2D, GL_R16, 1, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST, GL_NEAREST)
    
    textureDict['filteredDepth'] = createTexture(textureDict['filteredDepth'], GL_TEXTURE_2D, GL_R32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    
    textureDict['xyLUT'] = createTexture(textureDict['xyLUT'], GL_TEXTURE_2D, GL_RG32F, 1, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST, GL_NEAREST)

    textureDict['lastDepth'] = createTexture(textureDict['lastDepth'], GL_TEXTURE_2D, GL_R32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    textureDict['nextDepth'] = createTexture(textureDict['nextDepth'], GL_TEXTURE_2D, GL_R32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    
    textureDict['mappingC2D'] = createTexture(textureDict['mappingC2D'], GL_TEXTURE_2D, GL_RG16, numLevels, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), 1, GL_NEAREST, GL_NEAREST)
    textureDict['mappingD2C'] = createTexture(textureDict['mappingD2C'], GL_TEXTURE_2D, GL_RG16, numLevels, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), 1, GL_NEAREST, GL_NEAREST)

    textureDict['refVertex'] = createTexture(textureDict['refVertex'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    textureDict['virtualVertex'] = createTexture(textureDict['virtualVertex'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    
    textureDict['refNormal'] = createTexture(textureDict['refNormal'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    textureDict['virtualNormal'] = createTexture(textureDict['virtualNormal'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)

    textureDict['volume'] = createTexture(textureDict['volume'], GL_TEXTURE_3D, GL_RG16F, 1, fusionConfig['volSize'][0], fusionConfig['volSize'][1], fusionConfig['volSize'][2], GL_NEAREST, GL_NEAREST)

    textureDict['tracking'] = createTexture(textureDict['tracking'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    #nextFlowMap
    #textureList[5] = createTexture(textureList[5], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #sparseFlowMap
    #textureList[6] = createTexture(textureList[6], GL_TEXTURE_2D, GL_RGBA32F, maxLevels, int(width / 4), int(height / 4), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #densificationFlowMap
    #textureList[7] = createTexture(textureList[7], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

 
    

	# Allocate the immutable GPU memory storage -more efficient than mutable memory if you are not going to change image size after creation

    return textureDict

def createXYLUT(playback, textureDict, cameraConfig):

    xyTable = np.zeros((cameraConfig['depthHeight'], cameraConfig['depthWidth'], 2), dtype = "float")

    for x in range(cameraConfig['depthHeight']):
        for y in range(cameraConfig['depthWidth']):
            point = float(x), float(y)
            converted = playback.calibration.convert_2d_to_3d(point, 1.0, pyk4a.CalibrationType.DEPTH, pyk4a.CalibrationType.DEPTH)
            if np.isnan(converted).any():
                xyTable[x, y] = -1, -1  
            else:
                xyTable[x, y] = converted[0], converted[1]  
    
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, textureDict['xyLUT'])
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), GL_RG, GL_FLOAT, xyTable)

def bilateralFilter(shaderDict, textureDict, cameraConfig):
    glUseProgram(shaderDict['bilateralFilterShader'])
    # set logic for using filtered or unfiltered TODO

    glBindImageTexture(0, textureDict['rawDepth'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R16UI) 
    glBindImageTexture(1, textureDict['lastDepth'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F) 
    glBindImageTexture(2, textureDict['filteredDepth'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F) 

    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "depthScale"), 0.001)
    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "sigma"), 10.0)
    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "bSigma"), 0.05)

    compWidth = int((cameraConfig['depthWidth']/32.0)+0.5)
    compHeight = int((cameraConfig['depthHeight']/32.0)+0.5)

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

def depthToVertex(shaderDict, textureDict, cameraConfig, fusionConfig):

    glUseProgram(shaderDict['depthToVertexShader'])

    glBindImageTexture(0, textureDict['filteredDepth'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F) 
    glBindImageTexture(1, textureDict['xyLUT'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F) 
    glBindImageTexture(2, textureDict['refVertex'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    glUniform1f(glGetUniformLocation(shaderDict['depthToVertexShader'], "minDepth"), fusionConfig['nearPlane'])
    glUniform1f(glGetUniformLocation(shaderDict['depthToVertexShader'], "maxDepth"), fusionConfig['farPlane'])
    glUniform2f(glGetUniformLocation(shaderDict['depthToVertexShader'], "bottomLeft"), 0.0, 0.0)
    glUniform2f(glGetUniformLocation(shaderDict['depthToVertexShader'], "topRight"), 640.0, 576.0)
    glUniformMatrix4fv(glGetUniformLocation(shaderDict['depthToVertexShader'], "invK"), 1, False, glm.value_ptr(cameraConfig['invK']))

    compWidth = int((cameraConfig['depthWidth']/32.0)+0.5)
    compHeight = int((cameraConfig['depthHeight']/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def vertexToNormal(shaderDict, textureDict, cameraConfig):

    glUseProgram(shaderDict['vertexToNormalShader'])

    glBindImageTexture(0, textureDict['refVertex'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['refNormal'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    compWidth = int((cameraConfig['depthWidth']/32.0)+0.5)
    compHeight = int((cameraConfig['depthHeight']/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def mipmapTextures(textureDict):

    glBindTexture(GL_TEXTURE_2D, textureDict['filteredDepth'])
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

    glBindTexture(GL_TEXTURE_2D, textureDict['refVertex'])
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)

    glBindTexture(GL_TEXTURE_2D, textureDict['refNormal'])
    glGenerateMipmap(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, 0)    

def p2pTrack(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, level):
    glUseProgram(shaderDict['trackP2PShader'])

    glBindImageTexture(0, textureDict['refVertex'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['refNormal'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)

    glBindImageTexture(2, textureDict['virtualVertex'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(3, textureDict['virtualNormal'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 

    glBindImageTexture(4, textureDict['tracking'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8) 

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['p2pReduction'])

    glUniform1i(glGetUniformLocation(shaderDict['trackP2PShader'], "mip"), level)

    #invPose = glm.inverse(currPose)
    #view = cameraConfig['K'] * invPose

    glUniformMatrix4fv(glGetUniformLocation(shaderDict['trackP2PShader'], "K"), 1, False, glm.value_ptr(cameraConfig['K']))
    #glUniformMatrix4fv(glGetUniformLocation(shaderDict['trackP2PShader'], "pose"), 1, False, glm.value_ptr(currPose))

    glUniform1f(glGetUniformLocation(shaderDict['trackP2PShader'], "distThresh"), fusionConfig['distThresh'])
    glUniform1f(glGetUniformLocation(shaderDict['trackP2PShader'], "normThresh"), fusionConfig['normThresh'])

    compWidth = int((cameraConfig['depthWidth']/32.0)+0.5)
    compHeight = int((cameraConfig['depthHeight']/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def p2pReduce(shaderDict, bufferDict, cameraConfig, level):
    glUseProgram(shaderDict['reduceP2PShader'])

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['p2pReduction'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['p2pRedOut'])

    glUniform2i(glGetUniformLocation(shaderDict['reduceP2PShader'], "imSize"), int(cameraConfig['depthWidth'] >> level), int(cameraConfig['depthHeight'] >> level))

    glDispatchCompute(4 >> level, 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def getReductionP2P(bufferDict, level):        

    #sTime = time.perf_counter()

    #void_ptr = ctypes.c_void_p(ctypes.addressof(float32_data))

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['p2pRedOut'])
    #glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (640 >> level) * 4, void_ptr)
    tempData = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, (32 * (4 >> level)) * 4)
    # level 2 = 160
    # level 1 = 320
    # level 0 = 640
    #  
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    #eTime = time.perf_counter()
    #print((eTime - sTime) * 1000)

    #reductionData = np.ctypeslib.as_array(ctypes.cast(void_ptr, ctypes.POINTER(ctypes.c_float)), (640 >> level,))
    reductionData = np.frombuffer(tempData, dtype=np.float32)


    #   vector b
	# 	| 1 |
	# 	| 2 |
	# 	| 3 |
	# 	| 4 |
	# 	| 5 |
	# 	| 6 |

	# 	and
	# 	matrix a
	# 	| 7  | 8  | 9  | 10 | 11 | 12 |
	# 	| 8  | 13 | 14 | 15 | 16 | 17 |
	# 	| 9  | 14 | 18 | 19 | 20 | 21 |
	# 	| 10 | 15 | 19 | 22 | 23 | 24 |
	# 	| 11 | 16 | 20 | 23 | 25 | 26 |
	# 	| 12 | 17 | 21 | 24 | 26 | 27 |

	# 	AE = sqrt( [0] / [28] )
	# 	count = [28]

    vecb = np.zeros((6, 1), dtype='double')
    matA = np.zeros((6, 6), dtype='double')  

    for row in range(1, (4 >> level), 1):
        for col in range(0, 32, 1):
            reductionData[col] += reductionData[col + (row * 32)]

    for i in range(1, 7, 1):
        vecb[i - 1] = reductionData[i]

    shift = 6    
    for i in range(0, 6, 1):
        for j in range(i, 6, 1):
            shift += 1 # check this offset
            value = reductionData[shift]
            matA[j][i] = matA[i][j] = value
    
    AE = np.sqrt(reductionData[0] / reductionData[28])
    icpCount = reductionData[28]

    return matA, vecb, AE, icpCount

def solveP2P(shaderDict, bufferDict, level):
    glUseProgram(shaderDict['LDLTShader'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['p2pRedOut'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['poseBuffer'])

    glUniform1i(glGetUniformLocation(shaderDict['LDLTShader'], "mip"), level)

    glDispatchCompute(1, 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)


    # glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['poseBuffer'])
    # tempData = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16 * 4)
    # glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    # reductionData = np.frombuffer(tempData, dtype=np.float32)

    # return reductionData


def raycastVolume(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose):
    glUseProgram(shaderDict['raycastVolumeShader'])

    glBindImageTexture(0, textureDict['volume'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG16F) 
    glBindImageTexture(1, textureDict['virtualVertex'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 
    glBindImageTexture(2, textureDict['virtualNormal'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "nearPlane"), fusionConfig['nearPlane'])
    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "farPlane"), fusionConfig['farPlane'])
    
    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "volDim"), fusionConfig['volDim'][0])
    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "volSize"), fusionConfig['volSize'][0])

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])


    #view = currPose * cameraConfig['invK']
    #print(view)

    glUniformMatrix4fv(glGetUniformLocation(shaderDict['raycastVolumeShader'], "invK"), 1, False, glm.value_ptr(cameraConfig['invK']))
    glUniform3f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "initOffset"), fusionConfig['initOffset'][0], fusionConfig['initOffset'][1], fusionConfig['initOffset'][2])

    step = np.max(fusionConfig['volDim']) / np.max(fusionConfig['volSize'])
    largeStep = 0.375 # dont know why
    #dMin = -fusionConfig['volDim'][0] / 20.0
    #dMax = fusionConfig['volDim'][0] / 10.0
    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "step"), step)
    glUniform1f(glGetUniformLocation(shaderDict['raycastVolumeShader'], "largeStep"), largeStep)

    compWidth = int((cameraConfig['depthWidth']/32.0)+0.5) 
    compHeight = int((cameraConfig['depthHeight']/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def integrateVolume(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag, fusionTypeFlag):
    glUseProgram(shaderDict['integrateVolumeShader'])

    glBindImageTexture(0, textureDict['volume'], 0, GL_FALSE, 0, GL_READ_WRITE, GL_RG16F) 
    glBindImageTexture(1, textureDict['refVertex'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(2, textureDict['tracking'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8) 

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])

    glUniform1i(glGetUniformLocation(shaderDict['integrateVolumeShader'], "integrateFlag"), integrateFlag)
    glUniform1i(glGetUniformLocation(shaderDict['integrateVolumeShader'], "resetFlag"), resetFlag)

    #invT = glm.inverse(currPose)

    #glUniformMatrix4fv(glGetUniformLocation(shaderDict['integrateVolumeShader'], "invT"), 1, False, glm.value_ptr(invT))
    glUniformMatrix4fv(glGetUniformLocation(shaderDict['integrateVolumeShader'], "K"), 1, False, glm.value_ptr(cameraConfig['K']))

    glUniform1i(glGetUniformLocation(shaderDict['integrateVolumeShader'], "p2p"), fusionTypeFlag == 0)
    glUniform1i(glGetUniformLocation(shaderDict['integrateVolumeShader'], "p2v"), fusionTypeFlag == 1)

    glUniform1f(glGetUniformLocation(shaderDict['integrateVolumeShader'], "volDim"), fusionConfig['volDim'][0])
    glUniform1f(glGetUniformLocation(shaderDict['integrateVolumeShader'], "volSize"), fusionConfig['volSize'][0])
    glUniform1f(glGetUniformLocation(shaderDict['integrateVolumeShader'], "maxWeight"), fusionConfig['maxWeight'])

    compWidth = int((fusionConfig['volSize'][0]/32.0)) 
    compHeight = int((fusionConfig['volSize'][1]/32.0))

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def resultToMatrix(result):
    # from https://github.com/g-truc/glm/tree/master/glm/gtx/euler_angles.inl

    c1 = math.cos(-result[3])
    c2 = math.cos(-result[4])
    c3 = math.cos(-result[5])
    s1 = math.sin(-result[3])
    s2 = math.sin(-result[4])
    s3 = math.sin(-result[5])

    delta = glm.mat4(
        c2 * c3,   -c1 * s3 + s1 * s2 * c3,    s1 * s3 + c1 * s2 * c3,   0, 
        c2 * s3,    c1 * c3 + s1 * s2 * s3,   -s1 * c3 + c1 * s2 * s3,   0,
       -s2,         s1 * c2,                   c1 * c2,                  0, 
        result[0],  result[1],                 result[2],                1.0
    )

    return delta

def twist(xi):

    M = glm.mat4(0.0,    xi[2], -xi[1], 0.0, 
    -xi[2],  0.0,    xi[0], 0.0,
    xi[1], -xi[0],  0.0,   0.0,
    xi[3],  xi[4],  xi[5], 0.0)
  
    return M

def runP2P(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag):

    raycastVolume(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose)
    T = currPose

    for level in range((np.size(fusionConfig['iters']) - 1), -1, -1):
        for iter in range(fusionConfig['iters'][level - 1]):
            
            p2pTrack(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, T, level)

            p2pReduce(shaderDict, bufferDict, cameraConfig, level)

           # sTime = time.perf_counter()
            solveP2P(shaderDict, bufferDict, level)

        #     A, b, AE, icpCount = getReductionP2P(bufferDict, level)
        #   #  print('level : ', level, ((time.perf_counter() - sTime) * 1000))
        #     if (icpCount > 0):
        #         try:
        #             result = linalg.solve(A, b)
        #             c, low = linalg.cho_factor(A)
        #             res2 = linalg.cho_solve((c, low), b)
        #             print('done')
        #             #result = linalg.lu_solve((lu, piv), b)
        #         except:
        #             result = np.zeros((6, 1), dtype='double')
        #             continue
                
        #         delta = resultToMatrix(result)
        #     #     #d = glm.mat4(delta)

        #         T = delta * T
        #         #print(AE, icpCount)

        #         #eTime = time.time()
        #         #print((eTime - sTime) * 1000)

        #         resNorm = linalg.norm(result)

        #         if (resNorm < 1e-5 and resNorm != 0):
        #             break

    currPose = T
    if integrateFlag == True or resetFlag == True:
        integrateVolume(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag, 0)

    return currPose

def trackP2V(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, tempT, level):
    glUseProgram(shaderDict['trackP2VShader'])

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_3D, textureDict['volume'])

    glBindImageTexture(0, textureDict['refVertex'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['refNormal'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)

    glBindImageTexture(2, textureDict['virtualNormal'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 
    glBindImageTexture(3, textureDict['tracking'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8) 

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['p2vReduction'])

    glUniformMatrix4fv(glGetUniformLocation(shaderDict['trackP2VShader'], "T"), 1, False, glm.value_ptr(tempT))
    glUniform3f(glGetUniformLocation(shaderDict['trackP2VShader'], "volDim"), fusionConfig['volDim'][0], fusionConfig['volDim'][1], fusionConfig['volDim'][2])
    glUniform3f(glGetUniformLocation(shaderDict['trackP2VShader'], "volSize"), fusionConfig['volSize'][0], fusionConfig['volSize'][1], fusionConfig['volSize'][2])
    glUniform1i(glGetUniformLocation(shaderDict['trackP2VShader'], "mip"), level)

    compWidth = int(((cameraConfig['depthWidth'] >> level) / 32.0)+0.5)
    compHeight = int(((cameraConfig['depthHeight'] >> level) /32.0)+0.5)
    #print(compWidth)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def reduceP2V(shaderDict, bufferDict, cameraConfig, level):
    glUseProgram(shaderDict['reduceP2VShader'])

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['p2vReduction'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['p2vRedOut'])

    glUniform2i(glGetUniformLocation(shaderDict['reduceP2VShader'], "imSize"), cameraConfig['depthWidth'] >> level, cameraConfig['depthHeight'] >> level)

    glDispatchCompute(8, 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def getReductionP2V(bufferDict):        
    #sTime = time.perf_counter()
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['p2vRedOut'])
    tempData = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 32 * 8 * 4)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    reductionData = np.frombuffer(tempData, dtype=np.float32)
    #eTime = time.perf_counter()
    #print((eTime - sTime) * 1000.0)
    #   vector b
	# 	| 1 |
	# 	| 2 |
	# 	| 3 |
	# 	| 4 |
	# 	| 5 |
	# 	| 6 |

	# 	and
	# 	matrix a
	# 	| 7  | 8  | 9  | 10 | 11 | 12 |
	# 	| 8  | 13 | 14 | 15 | 16 | 17 |
	# 	| 9  | 14 | 18 | 19 | 20 | 21 |
	# 	| 10 | 15 | 19 | 22 | 23 | 24 |
	# 	| 11 | 16 | 20 | 23 | 25 | 26 |
	# 	| 12 | 17 | 21 | 24 | 26 | 27 |

	# 	AE = sqrt( [0] / [28] )
	# 	count = [28]

    vecb = np.zeros((6, 1), dtype='double')
    matA = np.zeros((6, 6), dtype='double')

    for row in range(1, 7, 1):
        for col in range(0, 31, 1):
            reductionData[col] += reductionData[col + (row * 32)]

    for i in range(1, 7, 1):
        vecb[i - 1] = reductionData[i]

    shift = 6    
    for i in range(0, 6, 1):
        for j in range(i, 6, 1):
            shift += 1 # check this offset
            value = reductionData[shift]
            matA[j][i] = matA[i][j] = value

    AE = np.sqrt(reductionData[0] / reductionData[28])
    icpCount = reductionData[28]

    return matA, vecb, AE, icpCount

def runP2V(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag):


    result = np.zeros((6, 1), dtype='double')
    resultPrev = np.zeros((6, 1), dtype='double')

    T = currPose
    prevT = T

    for level in range((np.size(fusionConfig['iters']) - 1), -1, -1):
        for iter in range(fusionConfig['iters'][level - 1]):
            
            #sTime = time.perf_counter()
            
            tempTarr = linalg.expm(np.array(twist(result)))
            #eTime = time.perf_counter()

            # pyglm errors out with an invalid pointer on the jetson nx if we dont init all at once
            tempTmat = glm.mat4(
                tempTarr[0][0], tempTarr[0][1], tempTarr[0][2], tempTarr[0][3],
                tempTarr[1][0], tempTarr[1][1], tempTarr[1][2], tempTarr[1][3],
                tempTarr[2][0], tempTarr[2][1], tempTarr[2][2], tempTarr[2][3],
                tempTarr[3][0], tempTarr[3][1], tempTarr[3][2], tempTarr[3][3]
            )

            currT = tempTmat * T
            
            #print('level : ', level, (eTime - sTime) * 1000.0)

            trackP2V(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currT, level)


            reduceP2V(shaderDict, bufferDict, cameraConfig, level)

            A, b, AE, icpCount = getReductionP2V(bufferDict)

            if (icpCount > 0):

                scaling = 1.0

                if (np.max(A) != 0 and (1.0 / np.max(A)) > 0.0):
                    scaling = np.max(A)


                A *= scaling
                b *= scaling

                adjA = A + (iter * np.identity(6, dtype='double'))

                try:
                    result = result - linalg.solve(adjA, b)
                    #lu, piv = linalg.lu_factor(A)
                    #result = linalg.lu_solve((lu, piv), b)
                except:
                    result = np.zeros((6, 1), dtype='double')
                    continue

                change = result - resultPrev
                cNorm = linalg.norm(change)

                resultPrev = result

                if (cNorm < 1e-4 and AE != 0):
                    break

    if (np.isnan(result).any()):
        result = np.zeros((6, 1), dtype='double')

    lnpa2 = linalg.expm(np.array(twist(result)))

    glnpa2 = glm.mat4(
        lnpa2[0][0], lnpa2[0][1], lnpa2[0][2], lnpa2[0][3],
        lnpa2[1][0], lnpa2[1][1], lnpa2[1][2], lnpa2[1][3],
        lnpa2[2][0], lnpa2[2][1], lnpa2[2][2], lnpa2[2][3],
        lnpa2[3][0], lnpa2[3][1], lnpa2[3][2], lnpa2[3][3]
    )

    currPose = glnpa2 * T 
                    
    if integrateFlag == True or resetFlag == True:
        integrateVolume(shaderDict, textureDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag, 1)

    return currPose

def reset(textureDict, bufferDict, cameraConfig, fusionConfig, clickedPoint3D):
    generateTextures(textureDict, cameraConfig, fusionConfig)
    #generateBuffers(bufferDict, cameraConfig)

    currPose = glm.mat4(1.0)
    currPose = glm.translate(glm.mat4(1.0), glm.vec3(-clickedPoint3D[0] + fusionConfig['volDim'][0] / 2.0, -clickedPoint3D[1] + fusionConfig['volDim'][0] / 2.0, -clickedPoint3D[2] + fusionConfig['volDim'][0] / 2.0))

    #currPose[3,0] = (fusionConfig['volDim'][0] / 2.0) - clickedPoint3D[0]
    #currPose[3,1] = (fusionConfig['volDim'][1] / 2.0) - clickedPoint3D[1]
    #currPose[3,2] = (fusionConfig['volDim'][2] / 2.0) - clickedPoint3D[2]

    integrateFlag = 0
    resetFlag = 1

    return currPose, integrateFlag, resetFlag

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
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "isYFlip"), 1)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderType"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderOptions"), opts)
    glUniform2f(glGetUniformLocation(shaderDict['renderShader'], "depthRange"), 0.0, 5.0)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)

    #render normals
    opts = 0 << 0 | 1 << 1 | 0 << 2 | 0 << 3 | 0 << 4
    xpos = w / 3.0
    glViewport(int(xpos), int(ypos), int(xwidth),h)
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textureDict['virtualNormal'])
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "isYFlip"), 1)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderType"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderOptions"), opts)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)   

    #render color
    opts = 0 << 0 | 0 << 1 | 1 << 2 | 0 << 3 | 0 << 4
    xpos = 2 * w / 3.0
    glViewport(int(xpos), int(ypos), int(xwidth),h)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, textureDict['virtualVertex'])
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "isYFlip"), 1)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderType"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderOptions"), opts)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)   

def main():

    useLiveKinect = True   

    #gc.disable()

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

    screenVertex_shader = (Path(__file__).parent / 'shaders/ScreenQuad.vert').read_text()

    screenFragment_shader = (Path(__file__).parent / 'shaders/ScreenQuad.frag').read_text()

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

    integrate_shader = (Path(__file__).parent / 'shaders/integrate.comp').read_text()
    integrateShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(integrate_shader, GL_COMPUTE_SHADER))

    trackP2P_shader = (Path(__file__).parent / 'shaders/p2pTrack.comp').read_text()
    trackP2PShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(trackP2P_shader, GL_COMPUTE_SHADER))

    reduceP2P_shader = (Path(__file__).parent / 'shaders/p2pReduce.comp').read_text()
    reduceP2PShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(reduceP2P_shader, GL_COMPUTE_SHADER))

    trackP2V_shader = (Path(__file__).parent / 'shaders/p2vTrack.comp').read_text()
    trackP2VShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(trackP2V_shader, GL_COMPUTE_SHADER))

    reduceP2V_shader = (Path(__file__).parent / 'shaders/p2vReduce.comp').read_text()
    reduceP2VShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(reduceP2V_shader, GL_COMPUTE_SHADER))

    LDLT_shader = (Path(__file__).parent / 'shaders/LDLT.comp').read_text()
    LDLTShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(LDLT_shader, GL_COMPUTE_SHADER))

    if useLiveKinect == False:
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


    #playback.configuration["color_format"] == ImageFormat.COLOR_MJPG
    #if k4a.configuration["depth_mode"] == pyk4a.DepthMode.NFOV_UNBINNED:
    #    print("hello")

    shaderDict = {
        'renderShader' : renderShader,
        'bilateralFilterShader' : bilateralFilterShader,
        'alignDepthColorShader' : alignDepthColorShader,
        'depthToVertexShader' : depthToVertexShader,
        'vertexToNormalShader' : vertexToNormalShader,
        'raycastVolumeShader' : raycastShader,
        'integrateVolumeShader' : integrateShader,
        'trackP2PShader' : trackP2PShader,
        'reduceP2PShader' : reduceP2PShader,
        'trackP2VShader' : trackP2VShader,
        'reduceP2VShader' : reduceP2VShader,
        'LDLTShader' : LDLTShader
    }

    bufferDict = {
        'p2pReduction' : -1,
        'p2pRedOut' : -1,
        'p2vReduction' : -1,
        'p2vRedOut' : -1,
        'test' : -1,
        'outBuf' : -1,
        'poseBuffer' : -1
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
#        'iters' : (2, 5, 10),

    fusionConfig = {
        'volSize' : (128, 128, 128),
        'volDim' : (1.0, 1.0, 1.0),
        'iters' : (2, 5, 10),
        'initOffset' : (0, 0, 0),
        'maxWeight' : 100.0,
        'distThresh' : 0.05,
        'normThresh' : 0.9,
        'nearPlane' : 0.1,
        'farPlane' : 1.0
    }

    cameraConfig = {
        'depthWidth' : 640,
        'depthHeight' : 576,
        'colorWidth' : 1920,
        'colorHeight' : 1080,
        'depthScale' : 0.001,
        'K' : K,
        'invK' : invK
    }

    textureDict = generateTextures(textureDict, cameraConfig, fusionConfig)
    bufferDict = generateBuffers(bufferDict, cameraConfig)

    colorMat = np.zeros((cameraConfig['colorHeight'], cameraConfig['colorWidth'], 3), dtype = "uint8")
    useColorMat = False 
    integrateFlag = True
    resetFlag = True
    initPose = glm.mat4()
    initPose[3,0] = fusionConfig['volDim'][0] / 2.0
    initPose[3,1] = fusionConfig['volDim'][1] / 2.0
    initPose[3,2] = 0


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['poseBuffer'])
    glBufferData(GL_SHADER_STORAGE_BUFFER, 16 * 4, glm.value_ptr(initPose), GL_DYNAMIC_COPY)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


    mouseX, mouseY = 0, 0
    clickedPoint3D = glm.vec4(fusionConfig['volDim'][0] / 2.0, fusionConfig['volDim'][1] / 2.0, 0, 0)
    sliderDim = fusionConfig['volDim'][0]

    #[32 64 128 256 512]
    currentSize = math.log2(fusionConfig['volSize'][0]) - 5
    volumeStatsChanged = False

    currPose = initPose


    # aa = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=torch.float32, device=torch.device('cuda'))
    # bb = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=torch.device('cuda'))

    # #setup pycuda gl interop needs to be after openGL is init
    # import pycuda.gl.autoinit
    # import pycuda.gl
    # cuda_gl = pycuda.gl
    # cuda_driver = pycuda.driver
    # from pycuda.compiler import SourceModule
    # import pycuda 
    
    # pycuda_source_ssbo = cuda_gl.RegisteredBuffer(int(bufferDict['test']), cuda_gl.graphics_map_flags.NONE)

    # sm = SourceModule("""
    #     __global__ void simpleCopy(float *inputArray, float *outputArray) {
    #             unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;

    #             outputArray[x] = inputArray[x];
    #             inputArray[x] = 8008.135f;
    #     }    
    # """)

    # cuda_function = sm.get_function("simpleCopy")

    # mappingObj = pycuda_source_ssbo.map()
    # data, size = mappingObj.device_ptr_and_size()

    # cuda_function(np.intp(aa.data_ptr()), np.intp(data), block=(8, 1, 1))

    # mappingObj.unmap()

    # glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['test'])
    # tee = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 32)
    # glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    # teeData = np.frombuffer(tee, dtype=np.float32)
    # print(teeData)

    # modTensor = aa.cpu().data.numpy()
    # print(modTensor)

    

    #fusionConfig['initOffset'] = (initPose[3,0], initPose[3,1], initPose[3,2])


    # LUTs
    #createXYLUT(k4a, textureDict, cameraConfig) <-- bug in this

    while not glfw.window_should_close(window):

        glfw.poll_events()
        impl.process_inputs()
        imgui.new_frame()
        

        sTime = time.perf_counter()

        try:
            if useLiveKinect == False:
                capture = k4a.get_next_capture()
            else:    
                capture = k4a.get_capture()
            if capture.color is not None:
                if useLiveKinect == False:
                    if k4a.configuration["color_format"] == ImageFormat.COLOR_MJPG:
                        colorMat = cv2.imdecode(capture.color, cv2.IMREAD_COLOR)
                        useColorMat = True

                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_2D, textureDict['lastColor'])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), (GL_RGB, GL_RGBA)[useLiveKinect], GL_UNSIGNED_BYTE, (capture.color, colorMat)[useColorMat] )

            if capture.depth is not None:
                glActiveTexture(GL_TEXTURE1)
                glBindTexture(GL_TEXTURE_2D, textureDict['rawDepth'])
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), GL_RED, GL_UNSIGNED_SHORT, capture.depth)

        except EOFError:
            break

        # #smallMat = cv2.pyrDown(colorMat)
        # start_time = time.time()
        # rotMat = cv2.flip(colorMat, 0)

        # pil_image = Image.fromarray(rotMat).convert('RGB')
        # image = transform(pil_image)
        

        # image = image.unsqueeze(0).to(device)
        # end_time = time.time()
        # print((end_time - start_time) * 1000.0)

        # with torch.no_grad():
        #     outputs = model(image)

        # output_image = utils.draw_keypoints(outputs, rotMat)
        # cv2.imshow('Face detection frame', output_image)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break





        bilateralFilter(shaderDict, textureDict, cameraConfig)
        depthToVertex(shaderDict, textureDict, cameraConfig, fusionConfig)
        #alignDepthColor(alignDepthColorShader, textureDict, colorWidth, colorHeight, depthWidth, depthHeight)
        vertexToNormal(shaderDict, textureDict, cameraConfig)

        mipmapTextures(textureDict)

        currPose = runP2P(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag)
        #currPose = runP2V(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag)
        eTime = time.perf_counter()
        print((eTime-sTime) * 1000)
        if resetFlag == True:
            resetFlag = False
            integrateFlag = True


        imgui.begin("Menu", True)
        if imgui.button("Reset"):
            fusionConfig['volSize'] = (1 << (currentSize + 5), 1 << (currentSize + 5), 1 << (currentSize + 5))
            fusionConfig['volDim'] = (sliderDim, sliderDim, sliderDim)
            currPose, integrateFlag, resetFlag = reset(textureDict, bufferDict, cameraConfig, fusionConfig, clickedPoint3D)
            volumeStatsChanged = False

        changedDim, sliderDim = imgui.slider_float("dim", sliderDim, min_value=0.01, max_value=5.0)

        clickedSize, currentSize = imgui.combo(
            "size", currentSize, ["32", "64", "128", "256", "512"]
        )
        
        if imgui.is_mouse_clicked():
            if not imgui.is_any_item_active():
                mouseX, mouseY = imgui.get_mouse_pos()
                w, h = glfw.get_framebuffer_size(window)
                xPos = ((mouseX % int(w / 3)) / (w / 3) * cameraConfig['depthWidth'])
                yPos = (mouseY / (h)) * cameraConfig['depthHeight']
                clickedDepth = capture.depth[int(yPos+0.5), int(xPos+0.5)] * cameraConfig['depthScale']
                clickedPoint3D = clickedDepth * (cameraConfig['invK'] * glm.vec4(xPos, yPos, 1.0, 0.0))
                volumeStatsChanged = True
             



        if changedDim or clickedSize:
            volumeStatsChanged = True

        imgui.end()

        render(VAO, window, shaderDict, textureDict)

        imgui.render()

        impl.render(imgui.get_draw_data())

        glfw.swap_buffers(window)        


    glfw.terminate()
    if useLiveKinect == True:
        k4a.stop()

    


if __name__ == "__main__":
    main()    