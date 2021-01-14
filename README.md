# pyglFlow
An openGL GLSL implementation of Stuff TBD Using a python front end

## Installation

Tested running on win10 with python 3.8 x64

```shell
$ pip install git+https://github.com/philipNoonan/pyglFlow.git
```

## Using pyglFlow

```
mkdir data
```

Place any video you want to play with into the /data folder.

```
$ pyglFlow
```

This opens a black window, with a menu allowing you to choose between Use Camera, and Select Video File. Either option prompts a new drop down list of camera IDs (click to choose 0 for default) or available video files found in the /data directory.
The sliders change which frame is rendered in either the R,G,B channels in the middle window. The 'run filter' option runs a simple gradient compute shader over the latest frame from the chosen video source.
