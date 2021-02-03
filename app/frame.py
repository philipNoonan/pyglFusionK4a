import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import glm
import numpy as np

import graphics

def generateFrameBuffers(fboDict, textureDict, cameraConfig):

    indexTexture = [textureDict['indexMap']]
    fboDict['indexMap'] = graphics.createFrameBuffer(fboDict['indexMap'], indexTexture, cameraConfig['depthWidth'] * 4, cameraConfig['depthHeight'] * 4)
    vfTextures = [textureDict['virtualVertex'], textureDict['virtualNormal'], textureDict['virtualDepth'], textureDict['virtualColor']]
    fboDict['virtualFrame'] = graphics.createFrameBuffer(fboDict['virtualFrame'], vfTextures, cameraConfig['depthWidth'], cameraConfig['depthHeight'])

    return fboDict

def generateTextures(textureDict, cameraConfig, fusionConfig):

    numLevels = np.size(fusionConfig['iters'])
    #lastColor
    textureDict['rawColor'] = graphics.createTexture(textureDict['rawColor'], GL_TEXTURE_2D, GL_RGBA8, 1, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

    textureDict['lastColor'] = graphics.createTexture(textureDict['lastColor'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    textureDict['nextColor'] = graphics.createTexture(textureDict['nextColor'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    
    textureDict['rawDepth'] = graphics.createTexture(textureDict['rawDepth'], GL_TEXTURE_2D, GL_R16, 1, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST, GL_NEAREST)
    
    textureDict['filteredDepth'] = graphics.createTexture(textureDict['filteredDepth'], GL_TEXTURE_2D, GL_R32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    
    textureDict['xyLUT'] = graphics.createTexture(textureDict['xyLUT'], GL_TEXTURE_2D, GL_RG32F, 1, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST, GL_NEAREST)

    textureDict['lastDepth'] = graphics.createTexture(textureDict['lastDepth'], GL_TEXTURE_2D, GL_R32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    textureDict['nextDepth'] = graphics.createTexture(textureDict['nextDepth'], GL_TEXTURE_2D, GL_R32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    
    textureDict['mappingC2D'] = graphics.createTexture(textureDict['mappingC2D'], GL_TEXTURE_2D, GL_RG16, numLevels, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), 1, GL_NEAREST, GL_NEAREST)
    textureDict['mappingD2C'] = graphics.createTexture(textureDict['mappingD2C'], GL_TEXTURE_2D, GL_RG16, numLevels, int(cameraConfig['colorWidth']), int(cameraConfig['colorHeight']), 1, GL_NEAREST, GL_NEAREST)

    textureDict['refVertex'] = graphics.createTexture(textureDict['refVertex'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    textureDict['virtualVertex'] = graphics.createTexture(textureDict['virtualVertex'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    
    textureDict['refNormal'] = graphics.createTexture(textureDict['refNormal'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    textureDict['virtualNormal'] = graphics.createTexture(textureDict['virtualNormal'], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)

    textureDict['virtualDepth'] = graphics.createTexture(textureDict['virtualDepth'], GL_TEXTURE_2D, GL_R32F, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    textureDict['virtualColor'] = graphics.createTexture(textureDict['virtualColor'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)

    textureDict['volume'] = graphics.createTexture(textureDict['volume'], GL_TEXTURE_3D, GL_RG16F, 1, fusionConfig['volSize'][0], fusionConfig['volSize'][1], fusionConfig['volSize'][2], GL_NEAREST, GL_NEAREST)

    textureDict['tracking'] = graphics.createTexture(textureDict['tracking'], GL_TEXTURE_2D, GL_RGBA8, numLevels, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), 1, GL_NEAREST_MIPMAP_NEAREST, GL_NEAREST)
    #nextFlowMap
    #textureList[5] = createTexture(textureList[5], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #sparseFlowMap
    #textureList[6] = createTexture(textureList[6], GL_TEXTURE_2D, GL_RGBA32F, maxLevels, int(width / 4), int(height / 4), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)
    #densificationFlowMap
    #textureList[7] = createTexture(textureList[7], GL_TEXTURE_2D, GL_RGBA32F, numLevels, int(width), int(height), 1, GL_LINEAR_MIPMAP_NEAREST, GL_LINEAR)

    textureDict['indexMap'] = graphics.createTexture(textureDict['indexMap'], GL_TEXTURE_2D, GL_R32F, 1, int(cameraConfig['depthWidth']) * 4, int(cameraConfig['depthHeight']) * 4, 1, GL_NEAREST, GL_NEAREST)

    

	# Allocate the immutable GPU memory storage -more efficient than mutable memory if you are not going to change image size after creation

    return textureDict

# def createXYLUT(playback, textureDict, cameraConfig):

#     xyTable = np.zeros((cameraConfig['depthHeight'], cameraConfig['depthWidth'], 2), dtype = "float")

#     for x in range(cameraConfig['depthHeight']):
#         for y in range(cameraConfig['depthWidth']):
#             point = float(x), float(y)
#             converted = playback.calibration.convert_2d_to_3d(point, 1.0, pyk4a.CalibrationType.DEPTH, pyk4a.CalibrationType.DEPTH)
#             if np.isnan(converted).any():
#                 xyTable[x, y] = -1, -1  
#             else:
#                 xyTable[x, y] = converted[0], converted[1]  
    
#     glActiveTexture(GL_TEXTURE0)
#     glBindTexture(GL_TEXTURE_2D, textureDict['xyLUT'])
#     glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(cameraConfig['depthWidth']), int(cameraConfig['depthHeight']), GL_RG, GL_FLOAT, xyTable)

def bilateralFilter(shaderDict, textureDict, cameraConfig):
    glUseProgram(shaderDict['bilateralFilterShader'])
    # set logic for using filtered or unfiltered TODO

    glBindImageTexture(0, textureDict['rawDepth'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R16UI) 
    glBindImageTexture(1, textureDict['lastDepth'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F) 
    glBindImageTexture(2, textureDict['filteredDepth'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F) 

    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "depthScale"), 0.001) # FIXME
    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "sigma"), 10.0)
    glUniform1f(glGetUniformLocation(shaderDict['bilateralFilterShader'], "bSigma"), 0.05)

    compWidth = int((cameraConfig['depthWidth']/32.0)+0.5)
    compHeight = int((cameraConfig['depthHeight']/32.0)+0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)    

def alignDepthColor(shaderDict, textureDict, cameraConfig, fusionConfig):
    glUseProgram(shaderDict['alignDepthColorShader'])

    level = 0
    glBindImageTexture(3, textureDict['mappingC2D'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16UI) 
    glBindImageTexture(4, textureDict['mappingD2C'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16UI) 
    
    glUniform1i(glGetUniformLocation(shaderDict['alignDepthColorShader'], "functionID"), 0)
    compWidth = int(((cameraConfig['depthWidth'] >> level) / 32.0) + 0.5)
    compHeight = int(((cameraConfig['depthHeight'] >> level) / 32.0) + 0.5)
    
    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

    glBindImageTexture(0, textureDict['refVertex'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['rawColor'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8) 
    glBindImageTexture(2, textureDict['lastColor'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8) 

    glBindImageTexture(3, textureDict['mappingC2D'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16UI) 
    glBindImageTexture(4, textureDict['mappingD2C'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16UI) 
    #print(cameraConfig["colK"][2, 0], cameraConfig["colK"][2, 1], cameraConfig["colK"][0, 0], cameraConfig["colK"][1, 1])

    glUniformMatrix4fv(glGetUniformLocation(shaderDict['alignDepthColorShader'], "d2c"), 1, False, glm.value_ptr(cameraConfig['d2c']))
    glUniform4f(glGetUniformLocation(shaderDict['alignDepthColorShader'], "cam"), 957.1860961914062, 553.4452514648438, 919.6978149414062, 919.4968872070312) # FIXME
    glUniform1i(glGetUniformLocation(shaderDict['alignDepthColorShader'], "functionID"), 1)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

    # for level in range((np.size(fusionConfig['iters']) - 1), -1, -1):
        
    #     glBindImageTexture(3, textureDict['mappingC2D'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16UI) 
    #     glBindImageTexture(4, textureDict['mappingD2C'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16UI) 

    #     glUniform1i(glGetUniformLocation(shaderDict['alignDepthColorShader'], "functionID"), 0)

    #     compWidth = int(((cameraConfig['depthWidth'] >> level) / 32.0) + 0.5)
    #     compHeight = int(((cameraConfig['depthHeight'] >> level) / 32.0) + 0.5)

    #     glDispatchCompute(compWidth, compHeight, 1)
    #     glMemoryBarrier(GL_ALL_BARRIER_BITS)
        
    #     colorCamPams = (
    #         cameraConfig["colK"][2, 0] / float(1 << (level)),
    #         cameraConfig["colK"][2, 1] / float(1 << (level)),
    #         cameraConfig["colK"][0, 0] / float(1 << (level)),
    #         cameraConfig["colK"][1, 1] / float(1 << (level))
    #     )

    #     glBindImageTexture(0, textureDict['refVertex'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    #     glBindImageTexture(1, textureDict['rawColor'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8) 
    #     glBindImageTexture(2, textureDict['lastColor'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8) 

    #     glBindImageTexture(3, textureDict['mappingC2D'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16UI) 
    #     glBindImageTexture(4, textureDict['mappingD2C'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RG16UI) 

    #     glUniformMatrix4fv(glGetUniformLocation(shaderDict['alignDepthColorShader'], "d2c"), 1, False, glm.value_ptr(cameraConfig['d2c']))
    #     glUniform4f(glGetUniformLocation(shaderDict['alignDepthColorShader'], "cam"), colorCamPams[0], colorCamPams[1], colorCamPams[2], colorCamPams[3])
    #     glUniform1i(glGetUniformLocation(shaderDict['alignDepthColorShader'], "functionID"), 1)

    #     glDispatchCompute(compWidth, compHeight, 1)
    #     glMemoryBarrier(GL_ALL_BARRIER_BITS)

def depthToVertex(shaderDict, textureDict, cameraConfig, fusionConfig):

    glUseProgram(shaderDict['depthToVertexShader'])

    glBindImageTexture(0, textureDict['filteredDepth'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F) 
    glBindImageTexture(1, textureDict['xyLUT'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RG32F) 
    glBindImageTexture(2, textureDict['refVertex'], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 

    glUniform1f(glGetUniformLocation(shaderDict['depthToVertexShader'], "minDepth"), fusionConfig['nearPlane'])
    glUniform1f(glGetUniformLocation(shaderDict['depthToVertexShader'], "maxDepth"), fusionConfig['farPlane'])
    glUniform2f(glGetUniformLocation(shaderDict['depthToVertexShader'], "bottomLeft"), 0.0, 0.0)
    glUniform2f(glGetUniformLocation(shaderDict['depthToVertexShader'], "topRight"), 640.0, 576.0) # FIXME
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


def generateBuffers(bufferDict, cameraConfig, fusionConfig):
    
    p2pRedBufSize = cameraConfig['depthWidth'] * cameraConfig['depthHeight'] * 8 * 4 # 8 float32 per depth pixel for reduction struct
    p2pRedOutBufSize = 32 * 4 * 4 # 32 outs per local group, upto 4 local groups running on highest def layer
    p2vRedBufSize = cameraConfig['depthWidth'] * cameraConfig['depthHeight'] * 9 * 4 # 9 float32 per depth pixel for reduction struct
    p2vRedOutBufSize = 32 * 8 * 4

    #flags = GL_DYNAMIC_STORAGE_BIT | 

    bufferDict['p2pReduction'] = graphics.createBuffer(bufferDict['p2pReduction'], GL_SHADER_STORAGE_BUFFER, p2pRedBufSize, 0)
    bufferDict['p2pRedOut'] = graphics.createBuffer(bufferDict['p2pRedOut'], GL_SHADER_STORAGE_BUFFER, p2pRedOutBufSize, 0)
    
    bufferDict['p2vReduction'] = graphics.createBuffer(bufferDict['p2vReduction'], GL_SHADER_STORAGE_BUFFER, p2vRedBufSize, 0)
    bufferDict['p2vRedOut'] = graphics.createBuffer(bufferDict['p2vRedOut'], GL_SHADER_STORAGE_BUFFER, p2vRedOutBufSize, 0)

    bufferDict['test'] = graphics.createBuffer(bufferDict['test'], GL_SHADER_STORAGE_BUFFER, 32, 0)
    bufferDict['outBuf'] = graphics.createBuffer(bufferDict['outBuf'], GL_SHADER_STORAGE_BUFFER, 36 * 4, 0)
    bufferDict['poseBuffer'] = graphics.createBuffer(bufferDict['poseBuffer'], GL_SHADER_STORAGE_BUFFER, (16 * 4 * 4) + (6 * 4), GL_DYNAMIC_STORAGE_BIT)

    bufferDict['globalMap0'] = graphics.createBuffer(bufferDict['globalMap0'], GL_SHADER_STORAGE_BUFFER, fusionConfig['maxMapSize'] * 4 * 4 * 4, 0)
    bufferDict['globalMap1'] = graphics.createBuffer(bufferDict['globalMap1'], GL_SHADER_STORAGE_BUFFER, fusionConfig['maxMapSize'] * 4 * 4 * 4, 0)

    bufferDict['atomic0'] = graphics.createBuffer(bufferDict['atomic0'], GL_ATOMIC_COUNTER_BUFFER, 4, GL_DYNAMIC_STORAGE_BIT)
    bufferDict['atomic1'] = graphics.createBuffer(bufferDict['atomic1'], GL_ATOMIC_COUNTER_BUFFER, 4, GL_DYNAMIC_STORAGE_BIT)

    return bufferDict