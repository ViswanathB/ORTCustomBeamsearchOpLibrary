# ORTCustomBeamsearchOpLibrary

## Prerequisites
Minimum requirements:
1. Windows SDK version 10.0.19041.0+
2. cmake, install cmake from https://cmake.org/install/
3. Python3 for using create_beam_search.py

### Note:
It contains a library file json.hpp from https://github.com/nlohmann/json/releases/tag/v3.10.5 for json processing and will be updated as needed.


## Introduction
This is an effort to provide equivalent of https://github.com/microsoft/onnxruntime/blob/master/docs/ContribOperators.md#com.microsoft.BeamSearch for customers that need custom beam search.

This currently only supports decoder models. [Create beam search script](./create_beam_search.py)(usage is documented within the script) generates config.json required to run the model.

Also, only the following parameters are provided as inputs to the post converted model: input ids, maximum length, vocab mask and prefix vocab mask. Other parameters needed for beam search are supplied from config.json that is generated during conversion. These are less likely to change duing deployment.

## Create custom op beam search for a decoder model
    
Custom op beam search for a decoder model can be created using:

    ```
    python create_beam_search.py -i decoder.onnx --vocab_size 50263 -o decoder_beamsearch.onnx --eos_token_id 50256 --pad_token_id 50262
    ```

There are some non mandatory options which can used or extended as needed.

## Build custom op
[python build file](build.py) is used to build the dll.

    ```
    python build.py
    ```

### Run test application with a converted model

Use above command to build the custom op and test application as following:

A [test application](./test_custom_beamsearch_op_library/) is provided in the repo. It is build automatically with no additional steps. Exe is build into "./test_custom_beamsearch_op_library/build/Debug/". It tests a simple case of batch_size = 1 and num_return_sequences = 1. Testing steps:

```
 cc_app.exe <decoder_beamsearch.onnx> <custom beam search op dll path> <test_data_file> <result_file>
```

1. decoder_beamsearch.onnx : post converted model path. The post converted model path looks like:
    ```
    decoder_beamsearch.onnx
        /model/decoder.onnx
        /model/config.json
    ```
    decoder_beamsearch.onnx has a attribute 'model_path' with path already set to '/model/decoder.onnx'

2. custom beam search op dll path would be : 'ORTCustomBeamsearchOpLibrary/build/Debug/custom_beamsearch_op_library_cpu.dll'

3. test_data_file: A sample test data file is provided. Each query of text is translated to post tokenization ids and prefix vocab mask and stored in a file. For instance, a query input has the following format:<br>
    line 1: "number of input ids" followed by "input ids..."<br>
    line 2: "number of vocab ids in prefix vocab mask that are 1s" followed by "prefix vocab mask ids"<br>

    ```
    52	50259 50260 50261 28446 428 5041 13 775 3088 10609 534 50166 3586 284 534 3227 3052 475 15223 2482 7603 326 262 734 5043 466 423 617 2443 393 12461 5400 326 673 1183 761 284 670 319 13 314 1183 1309 345 760 355 2582 355 340 338 3492 13	
    45	1757 4913 5030 5264 5302 5437 5689 5966 7212 8078 8518 11232 14087 15120 15251 15470 15768 16053 16709 16798 17960 18623 19161 20700 22568 23582 25824 25936 26154 26640 28056 28972 34687 36997 37406 38317 38579 39329 40276 40458 45447 45538 45644 47820 48608 
    ```

4. result_file: Where should the result be written. These are sequences post beam search


### Active Issues:
1. Python bind issue. the global env variable in pybind and C++ is different. When someone is trying to create an external session and an internal session, both have to share the same env. Current Onnxruntime code doesn't provide a way for this and thus, usage is limited to C++.

### Follow up, future releases
1. Output scores is not supported as 3rd output now, this should be added as contrib op supports this.
