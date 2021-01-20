# pyglFlow
An openGL GLSL implementation of Kinect Fusion using a python front end

## Installation

Tested running on win10 with python 3.8 x64

```shell
$ pip install git+https://github.com/philipNoonan/pyglFusionK4a.git
```

This may work, if not, then look at the setup.py and the imports and install things that sound like they should be installed. 

pycuda needs to be installed from source (i.e. not with pip install) with the following line added to the siteconf.py file created from running the configure.py script.

```
CUDA_ENABLE_GL = True
````


## Using pyglFusion


```
$ python app/pyglFusionK4A.py
```

Currently, you will have to edit the python file to select the recorded .mkv file. On success you will see a window showing the depth, normals, and color image from the kinect data.