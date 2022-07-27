# ORTCustomBeamsearchOpLibrary

## Prerequisites
Minimum requirements:
1. Windows SDK version 10.0.19041.0+
2. cmake, install cmake from https://cmake.org/install/
3. Python3 for using create_beam_search.py

### Note:
It contains a library file json.hpp from https://github.com/nlohmann/json/releases/tag/v3.10.5 for json processing and will be updated as needed.


## About the Custom OP
This is an effort to provide equivalent of https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.BeamSearch for customers that need custom beam search.

This currently only supports decoder models. [Create beam search script](./create_beam_search.py)(usage is documented within the script) generates config.json required to run the model.

Also, only the following parameters are provided as inputs to the post converted model: input ids, maximum length, vocab mask and prefix vocab mask. Other parameters needed for beam search are supplied from config.json that is generated during conversion. These are less likely to change duing deployment.

## Build 
[python build file](build.py) is used the build the dll. Follow the steps below:

## Integrate with onnxruntime

### Steps to build and run test application with a converted model
1. 

### Issues:
1. Python bind issue. the global env variable in pybind and C++ is different. When someone is trying to create an external session and an internal session, both have to share the same env. Current Onnxruntime code doesn't provide a way for this and thus, usage is limited to C++.

### Follow up, TODO
1. Output scores is not supported as 3rd output now, this should be added as contrib op supports this.
