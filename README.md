# pyglFusion
An openGL GLSL implementation of Kinect Fusion using a python front end. We try and do as little as possible in python, because speed. Memory copies from or to the GPU are limited to the bare minimum of uploading the camera image frames. 

Currently implemented:

1. Point to Point (p2p) fusion as in Newcombe et al. 2011 https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/ismar2011.pdf
2. Point to Volume (p2v) fusion as in Canelhas et al 2013 "SDF Tracker: A parallel algorithm for on-line pose estimation and scene reconstruction from depth images"
3. Splatter (splat) fusion as in Keller et al 2013 http://reality.cs.ucl.ac.uk/projects/kinect/keller13realtime.html

Splatter fusion currently uses atomic counters which may be slower than implementing optimized transform feedback shaders. 

## Installation

Tested running on win10 with python 3.8 x64, and the Nvidia Jetson Xavier NX (python 3.6)

```shell
$ pip install git+https://github.com/philipNoonan/pyglFusionK4a.git
```

This may work, if not, then just git clone the repo then look at the setup.py and the imports and install things that sound like they should be installed. 

The following is for experimental uses of pytorch and pycuda, they do not need to be used for the above fusion algorithms to be useable.

pycuda needs to be installed from source (i.e. not with pip install) with the following line added to the siteconf.py file created from running the configure.py script.

```
CUDA_ENABLE_GL = True
````


## Using pyglFusion


```
$ python app/pyglFusionK4A.py
```

Currently, you will have to edit the python file to select the recorded .mkv file. On success you will see a window showing the depth, normals, and color image from the kinect data.