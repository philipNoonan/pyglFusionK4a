import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import glm

import numpy as np

import frame

def p2pTrack(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, fusionType, level):
    glUseProgram(shaderDict['trackP2PShader'])

    glBindImageTexture(0, textureDict['refVertex'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['refNormal'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)

    glBindImageTexture(2, textureDict['virtualVertex'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(3, textureDict['virtualNormal'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 

    glBindImageTexture(4, textureDict['tracking'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8) 

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['p2pReduction'])

    glUniform1i(glGetUniformLocation(shaderDict['trackP2PShader'], "mip"), level)
    glUniform1i(glGetUniformLocation(shaderDict['trackP2PShader'], "fusionType"), fusionType)

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

    #print(matA)


    return matA, vecb, AE, icpCount

def solveP2P(shaderDict, bufferDict, fusionType, finalPass, level):
    glUseProgram(shaderDict['LDLTShader'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['p2pRedOut'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['poseBuffer'])

    glUniform1i(glGetUniformLocation(shaderDict['LDLTShader'], "mip"), level)
    glUniform1i(glGetUniformLocation(shaderDict['LDLTShader'], "fusionType"), fusionType)
    glUniform1i(glGetUniformLocation(shaderDict['LDLTShader'], "finalPass"), finalPass)

    glDispatchCompute(1, 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)


    # glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['poseBuffer'])
    # tempData = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16 * 4 * 4 + 6 * 4)
    # glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    # reductionData = np.frombuffer(tempData, dtype=np.float32)
    # print(reductionData)

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

    M = glm.mat4(0.0,    xi[2],  -xi[1],  0.0, 
               -xi[2],   0.0,     xi[0],  0.0,
                xi[1],  -xi[0],   0.0,    0.0,
                xi[3],   xi[4],   xi[5],  0.0)
  
    return M

def runP2P(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag):

    raycastVolume(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose)
    T = currPose

    for level in range((np.size(fusionConfig['iters']) - 1), -1, -1):
        for iter in range(fusionConfig['iters'][level - 1]):
            
            p2pTrack(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, 0, level)

            p2pReduce(shaderDict, bufferDict, cameraConfig, level)

            solveP2P(shaderDict, bufferDict, 0, 0, level)

    currPose = T
    if integrateFlag == True or resetFlag == True:
        integrateVolume(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag, 0)

    return currPose

def trackP2V(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, level):
    glUseProgram(shaderDict['trackP2VShader'])

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_3D, textureDict['volume'])

    glBindImageTexture(0, textureDict['refVertex'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(1, textureDict['refNormal'], level, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F)

    glBindImageTexture(2, textureDict['virtualNormal'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F) 
    glBindImageTexture(3, textureDict['tracking'], level, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA8) 

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['p2vReduction'])

    #glUniformMatrix4fv(glGetUniformLocation(shaderDict['trackP2VShader'], "T"), 1, False, glm.value_ptr(tempT))
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

    glUniform2i(glGetUniformLocation(shaderDict['reduceP2VShader'], "imSize"), int(cameraConfig['depthWidth'] >> level), int(cameraConfig['depthHeight'] >> level))

    glDispatchCompute(4 >> level, 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

def solveP2V(shaderDict, bufferDict, fusionType, finalPass, level, iter):
    glUseProgram(shaderDict['LDLTShader'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['p2vRedOut'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['poseBuffer'])

    glUniform1i(glGetUniformLocation(shaderDict['LDLTShader'], "mip"), level)
    glUniform1i(glGetUniformLocation(shaderDict['LDLTShader'], "iter"), iter)
    glUniform1i(glGetUniformLocation(shaderDict['LDLTShader'], "fusionType"), fusionType)
    glUniform1i(glGetUniformLocation(shaderDict['LDLTShader'], "finalPass"), finalPass)

    glDispatchCompute(1, 1, 1)
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


    #result = np.zeros((6, 1), dtype='double')
    #resultPrev = np.zeros((6, 1), dtype='double')
    finalPass = 0
    fusionType = 1

    #T = currPose
    #prevT = T

    for level in range((np.size(fusionConfig['iters']) - 1), -1, -1):
        for iter in range(fusionConfig['iters'][level]):
            #if (level == 0 and iter == (fusionConfig['iters'][0] - 1)):
            #    finalPass = 1
            
            #tempTarr = linalg.expm(np.array(twist(result)))

            glUseProgram(shaderDict['expm'])
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])
            glUniform1i(glGetUniformLocation(shaderDict['expm'], "finalPass"), finalPass)
            #glUniformMatrix4fv(glGetUniformLocation(shaderDict['expm'], "inputMat"), 1, False, glm.value_ptr(twist(result)))
            glDispatchCompute(1, 1, 1)
            glMemoryBarrier(GL_ALL_BARRIER_BITS)





            # pyglm errors out with an invalid pointer on the jetson nx if we dont init all at once
            # tempTmat = glm.mat4(
            #     tempTarr[0][0], tempTarr[0][1], tempTarr[0][2], tempTarr[0][3],
            #     tempTarr[1][0], tempTarr[1][1], tempTarr[1][2], tempTarr[1][3],
            #     tempTarr[2][0], tempTarr[2][1], tempTarr[2][2], tempTarr[2][3],
            #     tempTarr[3][0], tempTarr[3][1], tempTarr[3][2], tempTarr[3][3]
            # )

            # currT = tempTmat * T
            
            trackP2V(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, level)
            reduceP2V(shaderDict, bufferDict, cameraConfig, level)
            solveP2V(shaderDict, bufferDict, fusionType, finalPass, level, iter)

            # glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['poseBuffer'])
            # tempData = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4 * 4, 6 * 4)
            # glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            # reductionData = np.frombuffer(tempData, dtype=np.float32)
            # print(reductionData)

            #A, b, AE, icpCount = getReductionP2V(bufferDict)

            #print(b)
            #print(A)

            # if (icpCount > 0):

            #     scaling = 1.0

            #     if (np.max(A) != 0 and (1.0 / np.max(A)) > 0.0):
            #         scaling = np.max(A)


            #     A *= scaling
            #     b *= scaling

            #     adjA = A + (iter * np.identity(6, dtype='double'))

            #     try:
            #         result = result - linalg.solve(adjA, b)
            #         #lu, piv = linalg.lu_factor(A)
            #         #result = linalg.lu_solve((lu, piv), b)
            #     except:
            #         result = np.zeros((6, 1), dtype='double')
            #         continue

            #     change = result - resultPrev
            #     cNorm = linalg.norm(change)

            #     resultPrev = result

            #     if (cNorm < 1e-4 and AE != 0):
            #         break

    #if (np.isnan(result).any()):
    #    result = np.zeros((6, 1), dtype='double')

    finalPass = 1
    glUseProgram(shaderDict['expm'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])
    glUniform1i(glGetUniformLocation(shaderDict['expm'], "finalPass"), finalPass)
    glDispatchCompute(1, 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

    #lnpa2 = linalg.expm(np.array(twist(result)))

    #glnpa2 = glm.mat4(
    #    lnpa2[0][0], lnpa2[0][1], lnpa2[0][2], lnpa2[0][3],
    #    lnpa2[1][0], lnpa2[1][1], lnpa2[1][2], lnpa2[1][3],
    #    lnpa2[2][0], lnpa2[2][1], lnpa2[2][2], lnpa2[2][3],
    #    lnpa2[3][0], lnpa2[3][1], lnpa2[3][2], lnpa2[3][3]
    #)

    #currPose = glnpa2 * prevT 
                    
    if integrateFlag == True or resetFlag == True:
        integrateVolume(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag, 1)

    return currPose

def generateIndexMap(shaderDict, textureDict, bufferDict, fboDict, cameraConfig, fusionConfig, mapSize):
    glUseProgram(shaderDict['indexMapGeneration'])
    glEnable(GL_DEPTH_TEST)

    glBindFramebuffer(GL_FRAMEBUFFER, fboDict['indexMap'])
	
    glClearColor(-1.0, -1.0, -1.0, -1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, cameraConfig['depthWidth'] * 4, cameraConfig['depthHeight'] * 4)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['globalMap0'])

    glUniform2f(glGetUniformLocation(shaderDict['indexMapGeneration'], "imSize"), cameraConfig['depthWidth'], cameraConfig['depthHeight'])
    glUniform1f(glGetUniformLocation(shaderDict['indexMapGeneration'], "maxDepth"), fusionConfig['farPlane'])
    glUniform4f(glGetUniformLocation(shaderDict['indexMapGeneration'], "cam"), 320.0780029296875, 317.72021484375, 504.03192138671875, 504.2516784667969)

    glDrawArrays(GL_POINTS, 0, mapSize[0])    

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

def updateGlobalMap(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, mapSize, frameCount, firstFrame):
    glUseProgram(shaderDict['globalMapUpdate'])
    #sTime = time.perf_counter()

    
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, bufferDict['atomic0'])

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bufferDict['atomic0'])
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, mapSize)
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

    #print((time.perf_counter() - sTime) * 1000)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['globalMap0'])


    glUniformMatrix4fv(glGetUniformLocation(shaderDict['globalMapUpdate'], "K"), 1, False, glm.value_ptr(cameraConfig['K']))
    glUniform1i(glGetUniformLocation(shaderDict['globalMapUpdate'], "timestamp"), frameCount)
    glUniform1i(glGetUniformLocation(shaderDict['globalMapUpdate'], "firstFrame"), firstFrame)
    glUniform1f(glGetUniformLocation(shaderDict['globalMapUpdate'], "sigma"), fusionConfig['sigma'])
    glUniform1i(glGetUniformLocation(shaderDict['globalMapUpdate'], "maxMapSize"), fusionConfig['maxMapSize'])
    glUniform1f(glGetUniformLocation(shaderDict['globalMapUpdate'], "c_stable"), fusionConfig['c_stable'])

    glBindImageTexture(0, textureDict['indexMap'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_R32F) 
    glBindImageTexture(1, textureDict['refVertex'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(2, textureDict['refNormal'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F) 
    glBindImageTexture(3, textureDict['lastColor'], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA8) 

    compWidth = int((cameraConfig['depthWidth']/32.0) + 0.5) 
    compHeight = int((cameraConfig['depthHeight']/32.0) + 0.5)

    glDispatchCompute(compWidth, compHeight, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bufferDict['atomic0'])
    tempData = glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4)
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

    mapSize = np.array(np.frombuffer(tempData, dtype=np.uint32))

    return mapSize

def removeUnnecessaryPoints(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, mapSize, frameCount):

    glUseProgram(shaderDict['unnecessaryPointRemoval'])
    glUniform1i(glGetUniformLocation(shaderDict['unnecessaryPointRemoval'], "timestamp"), frameCount)
    glUniform1f(glGetUniformLocation(shaderDict['unnecessaryPointRemoval'], "c_stable"), fusionConfig['c_stable'])

    blank = np.array([0], dtype='uint32')
    glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 1, bufferDict['atomic1'])
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bufferDict['atomic1'])
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, blank)
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['globalMap0'])
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['globalMap1'])

    glDispatchCompute(int((mapSize[0] / 400) + 0.5), 1, 1)
    glMemoryBarrier(GL_ALL_BARRIER_BITS)

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bufferDict['atomic1'])
    tempData = glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4)
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

    mapSize = np.array(np.frombuffer(tempData, dtype=np.uint32))

    bufferDict['atomic0'], bufferDict['atomic1'] = bufferDict['atomic1'], bufferDict['atomic0']
    bufferDict['globalMap0'], bufferDict['globalMap1'] = bufferDict['globalMap1'], bufferDict['globalMap0']

    return mapSize, bufferDict

def genVirtualFrame(shaderDict, textureDict, bufferDict, fboDict, cameraConfig, fusionConfig, mapSize, frameCount):
    glUseProgram(shaderDict['surfaceSplatting'])

    glBindFramebuffer(GL_FRAMEBUFFER, fboDict['virtualFrame'])

    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, cameraConfig['depthWidth'], cameraConfig['depthHeight'])

    drawBuffs = [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2, GL_COLOR_ATTACHMENT3]

    glUniform2f(glGetUniformLocation(shaderDict['surfaceSplatting'], "imSize"), cameraConfig['depthWidth'], cameraConfig['depthHeight'])
    glUniform1f(glGetUniformLocation(shaderDict['surfaceSplatting'], "maxDepth"), fusionConfig['farPlane'])
    glUniform4f(glGetUniformLocation(shaderDict['surfaceSplatting'], "cam"), 320.0780029296875, 317.72021484375, 504.03192138671875, 504.2516784667969)
    glUniform1f(glGetUniformLocation(shaderDict['surfaceSplatting'], "c_stable"), fusionConfig['c_stable'])

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bufferDict['poseBuffer'])

    glEnable(GL_PROGRAM_POINT_SIZE)
    #glEnable(GL_POINT_SPRITE)

    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bufferDict['globalMap0']) 

    glDrawBuffers(4, drawBuffs)
    glDrawArrays(GL_POINTS, 0, mapSize[0])


    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    glDisable(GL_PROGRAM_POINT_SIZE)
    #glDisable(GL_POINT_SPRITE)

def runSplatter(shaderDict, textureDict, bufferDict, fboDict, cameraConfig, fusionConfig, mapSize, frameCount, integrateFlag, resetFlag):
    fusionType = 2
    finalPass = 0
    for level in range((np.size(fusionConfig['iters']) - 1), -1, -1):
        for iter in range(fusionConfig['iters'][level]):
            if (level == 0 and iter == (fusionConfig['iters'][0] - 1)):
                finalPass = 1
            p2pTrack(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, fusionType, level)
            p2pReduce(shaderDict, bufferDict, cameraConfig, level)
            solveP2P(shaderDict, bufferDict, fusionType, finalPass, level)
    


    generateIndexMap(shaderDict, textureDict, bufferDict, fboDict, cameraConfig, fusionConfig, mapSize)

    if (integrateFlag):
        mapSize = updateGlobalMap(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, mapSize, frameCount, resetFlag)
        mapSize, bufferDict = removeUnnecessaryPoints(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, mapSize, frameCount)


    genVirtualFrame(shaderDict, textureDict, bufferDict, fboDict, cameraConfig, fusionConfig, mapSize, frameCount)

    return mapSize

def reset(textureDict, bufferDict, cameraConfig, fusionConfig, clickedPoint3D):
    frame.generateTextures(textureDict, cameraConfig, fusionConfig)
    #generateBuffers(bufferDict, cameraConfig)

    currPose = glm.mat4(1.0)
    currPose = glm.translate(glm.mat4(1.0), glm.vec3(-clickedPoint3D[0] + fusionConfig['volDim'][0] / 2.0, -clickedPoint3D[1] + fusionConfig['volDim'][0] / 2.0, -clickedPoint3D[2] + fusionConfig['volDim'][0] / 2.0))

    #currPose[3,0] = (fusionConfig['volDim'][0] / 2.0) - clickedPoint3D[0]
    #currPose[3,1] = (fusionConfig['volDim'][1] / 2.0) - clickedPoint3D[1]
    #currPose[3,2] = (fusionConfig['volDim'][2] / 2.0) - clickedPoint3D[2]



    blankResult = np.array([0, 0, 0, 0, 0, 0], dtype='float32')
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['poseBuffer'])
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16 * 4, glm.value_ptr(currPose))
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4, 16 * 4, glm.value_ptr(glm.inverse(currPose)))
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4 * 2, 16 * 4, glm.value_ptr(glm.mat4(1.0)))
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4 * 3, 16 * 4, glm.value_ptr(glm.mat4(1.0)))
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4 * 4, 6 * 4, blankResult)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    initAtomicCount = np.array([0], dtype='uint32')
    mapSize = np.array([0], dtype='uint32')

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bufferDict['atomic0'])
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, initAtomicCount)
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bufferDict['atomic1'])
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, initAtomicCount)
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

    integrateFlag = 0
    resetFlag = 1

    return currPose, integrateFlag, resetFlag