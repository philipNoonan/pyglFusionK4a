import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import glm

import track
import frame
import graphics

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

    # Splatter
    globalMapUpdate_shader = (Path(__file__).parent / 'shaders/GlobalMapUpdate.comp').read_text()
    globalMapUpdateShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(globalMapUpdate_shader, GL_COMPUTE_SHADER))

    indexMapGenVert_shader = (Path(__file__).parent / 'shaders/IndexMapGeneration.vert').read_text()

    indexMapGenFrag_shader = (Path(__file__).parent / 'shaders/IndexMapGeneration.frag').read_text()

    IndexMapGenerationShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(indexMapGenVert_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(indexMapGenFrag_shader, GL_FRAGMENT_SHADER))

    SurfaceSplattingVert_shader = (Path(__file__).parent / 'shaders/SurfaceSplatting.vert').read_text()

    SurfaceSplattingFrag_shader = (Path(__file__).parent / 'shaders/SurfaceSplatting.frag').read_text()

    SurfaceSplattingShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(SurfaceSplattingVert_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(SurfaceSplattingFrag_shader, GL_FRAGMENT_SHADER))

    UnnecessaryPointRemoval_shader = (Path(__file__).parent / 'shaders/UnnecessaryPointRemoval.comp').read_text()
    UnnecessaryPointRemovalShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(UnnecessaryPointRemoval_shader, GL_COMPUTE_SHADER))

    # P2V 
    expm_shader = (Path(__file__).parent / 'shaders/expm.comp').read_text()
    expmShader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(expm_shader, GL_COMPUTE_SHADER))


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
        'LDLTShader' : LDLTShader,
        'globalMapUpdate' : globalMapUpdateShader,
        'indexMapGeneration' : IndexMapGenerationShader,
        'surfaceSplatting' : SurfaceSplattingShader,
        'unnecessaryPointRemoval' : UnnecessaryPointRemovalShader,
        'expm' : expmShader
    }

    bufferDict = {
        'p2pReduction' : -1,
        'p2pRedOut' : -1,
        'p2vReduction' : -1,
        'p2vRedOut' : -1,
        'test' : -1,
        'outBuf' : -1,
        'poseBuffer' : -1,
        'globalMap0' : -1,
        'globalMap1' : -1,
        'atomic0' : -1,
        'atomic1' : -1
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
        'refNormal' : -1,
        'virtualVertex' : -1,
        'virtualNormal' : -1,
        'virtualDepth' : -1,
        'virtualColor' : -1,
        'mappingC2D' : -1,
        'mappingD2C' : -1,
        'xyLUT' : -1, 
        'tracking' : -1,
        'volume' : -1,
        'indexMap' : -1
    }

    fboDict = {
        'indexMap' : -1,
        'virtualFrame' : -1
    }
#        'iters' : (2, 5, 10),

    fusionConfig = {
        'volSize' : (128, 128, 128),
        'volDim' : (1.0, 1.0, 1.0),
        'iters' : (2, 2, 2),
        'initOffset' : (0, 0, 0),
        'maxWeight' : 100.0,
        'distThresh' : 0.05,
        'normThresh' : 0.9,
        'nearPlane' : 0.1,
        'farPlane' : 4.0,
        'maxMapSize' : 5000000,
        'c_stable' : 10.0,
        'sigma' : 0.6
    }

    cameraConfig = {
        'depthWidth' : 640,
        'depthHeight' : 576,
        'colorWidth' : 1920,
        'colorHeight' : 1080,
        'd2c' : d2c,
        'c2d' : c2d,
        'depthScale' : 0.001,
        'K' : K,
        'invK' : invK,
        'colK' : colK
    }

    textureDict = frame.generateTextures(textureDict, cameraConfig, fusionConfig)
    bufferDict =  frame.generateBuffers(bufferDict, cameraConfig, fusionConfig)

    colorMat = np.zeros((cameraConfig['colorHeight'], cameraConfig['colorWidth'], 3), dtype = "uint8")
    useColorMat = False 
    integrateFlag = True
    resetFlag = True
    initPose = glm.mat4()
    initPose[3,0] = fusionConfig['volDim'][0] / 2.0
    initPose[3,1] = fusionConfig['volDim'][1] / 2.0
    initPose[3,2] = 0

    blankResult = np.array([0, 0, 0, 0, 0, 0], dtype='float32')


    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bufferDict['poseBuffer'])
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 16 * 4, glm.value_ptr(initPose))
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4, 16 * 4, glm.value_ptr(glm.inverse(initPose)))
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4 * 2, 16 * 4, glm.value_ptr(glm.mat4(1.0)))
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4 * 3, 16 * 4, glm.value_ptr(glm.mat4(1.0)))
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 16 * 4 * 4, 6 * 4, blankResult)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


    mouseX, mouseY = 0, 0
    clickedPoint3D = glm.vec4(fusionConfig['volDim'][0] / 2.0, fusionConfig['volDim'][1] / 2.0, 0, 0)
    sliderDim = fusionConfig['volDim'][0]

    #[32 64 128 256 512]
    currentSize = math.log2(fusionConfig['volSize'][0]) - 5
    volumeStatsChanged = False

    currPose = initPose

    # splatter stuff
    frameCount = 0
    fboDict = frame.generateFrameBuffers(fboDict, textureDict, cameraConfig )

    initAtomicCount = np.array([0], dtype='uint32')
    mapSize = np.array([0], dtype='uint32')

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bufferDict['atomic0'])
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, initAtomicCount)
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, bufferDict['atomic1'])
    glBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, initAtomicCount)
    glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, 0)

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
                glBindTexture(GL_TEXTURE_2D, textureDict['rawColor'])
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





        frame.bilateralFilter(shaderDict, textureDict, cameraConfig)
        frame.depthToVertex(shaderDict, textureDict, cameraConfig, fusionConfig)
        frame.alignDepthColor(shaderDict, textureDict, cameraConfig, fusionConfig)
        frame.vertexToNormal(shaderDict, textureDict, cameraConfig)

        frame.mipmapTextures(textureDict)



        #currPose = track.runP2P(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag)
        currPose = track.runP2V(shaderDict, textureDict, bufferDict, cameraConfig, fusionConfig, currPose, integrateFlag, resetFlag)
       
        #mapSize = track.runSplatter(shaderDict, textureDict, bufferDict, fboDict, cameraConfig, fusionConfig, mapSize, frameCount, integrateFlag, resetFlag)
        frameCount += 1

        if resetFlag == True:
            resetFlag = False
            integrateFlag = True


        imgui.begin("Menu", True)
        if imgui.button("Reset"):
            fusionConfig['volSize'] = (1 << (currentSize + 5), 1 << (currentSize + 5), 1 << (currentSize + 5))
            fusionConfig['volDim'] = (sliderDim, sliderDim, sliderDim)
            currPose, integrateFlag, resetFlag = track.reset(textureDict, bufferDict, cameraConfig, fusionConfig, clickedPoint3D)
            volumeStatsChanged = False

        if imgui.button("Integrate"):
            integrateFlag = not integrateFlag    
        imgui.same_line()
        imgui.checkbox("", integrateFlag)

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

        graphics.render(VAO, window, shaderDict, textureDict)

        imgui.render()

        impl.render(imgui.get_draw_data())

        eTime = time.perf_counter()

        #print((eTime-sTime) * 1000, mapSize[0])


        glfw.swap_buffers(window)        


    glfw.terminate()
    if useLiveKinect == True:
        k4a.stop()

    


if __name__ == "__main__":
    main()    