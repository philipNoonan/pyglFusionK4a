import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import glm

def createFrameBuffer(fbo, texList, width, height):

    if (fbo == -1):
        fbo = glGenFramebuffers(1)
    

    glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    depthTex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, depthTex)

    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, int(width), int(height), 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, None)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthTex, 0)
    for texNum in range(len(texList)):
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + texNum, GL_TEXTURE_2D, texList[texNum], 0)

    status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    if status != GL_FRAMEBUFFER_COMPLETE:
        print("framebuffer incomplete!")
        
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return fbo



def createBuffer(buffer, bufferType, size, usage):
    if buffer == -1:
        bufName = glGenBuffers(1)
    else:
        glDeleteBuffers(1, buffer)
        bufName = buffer
        bufName = glGenBuffers(1)

    glBindBuffer(bufferType, bufName)
    glBufferStorage(bufferType, size, None, usage)
    glBindBuffer(bufferType, 0)

    return bufName



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






def render(VAO, window, shaderDict, textureDict):

    glUseProgram(shaderDict['renderShader'])
    glClear(GL_COLOR_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)

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
    glBindTexture(GL_TEXTURE_2D, textureDict['lastColor'])
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "isYFlip"), 1)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderType"), 0)
    glUniform1i(glGetUniformLocation(shaderDict['renderShader'], "renderOptions"), opts)
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, None)   