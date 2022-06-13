// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <cstdint>
#include "beam_search.h"
#include <iostream>

using namespace std;

void SetBeamSearchOutputToZero(OrtKernelContext* context, Ort::CustomOpApi &ort, int batch_size, int seq_len) {
  //TODO Temp function to set the outputs of custom beam search OP to some value
  // so there is no error, this would be ideally set to 
  std::cout<<"Inside beam search, SetBeamSearchOutputToZero:"<<std::endl;
  const OrtValue* input_ids = ort.KernelContext_GetInput(context, 0);

  const OrtValue* num_ret_seqs = ort.KernelContext_GetInput(context, 4);
  const int* num_ret_seqs_data = ort.GetTensorData<int>(num_ret_seqs);
  const int num_seqs = num_ret_seqs_data[0];
  std::cout<<"num_seqs:"<<num_seqs<<std::endl;

  std::vector<int64_t> output1_dims;
  output1_dims.push_back(batch_size);
  output1_dims.push_back(num_seqs);
  output1_dims.push_back(seq_len);

  OrtValue* output1 = ort.KernelContext_GetOutput(context, 0, output1_dims.data(), output1_dims.size());
  int* out1 = ort.GetTensorMutableData<int>(output1);

  OrtTensorTypeAndShapeInfo* output_info1 = ort.GetTensorTypeAndShape(output1);
  std::vector<int64_t> tensor_shape = ort.GetTensorShape(output_info1);

  std::cout<<"Tensor shape of first output of custom bs OP"<<std::endl;
  for (int i=0;i<tensor_shape.size();i++){
    std::cout<<tensor_shape[i]<<",";
  }
  std::cout<<std::endl;

  int64_t size1 = ort.GetTensorShapeElementCount(output_info1);
  ort.ReleaseTensorTypeAndShapeInfo(output_info1);
  for (int64_t i = 0; i < size1-1; i++) {
    out1[i] = 10;
  }

  std::vector<int64_t> output2_dims;
  output2_dims.push_back(batch_size);
  output2_dims.push_back(num_seqs);

  OrtValue* output2 = ort.KernelContext_GetOutput(context, 1, output2_dims.data(), output2_dims.size());
  float* out2 = ort.GetTensorMutableData<float>(output2);

  OrtTensorTypeAndShapeInfo* output_info2 = ort.GetTensorTypeAndShape(output2);
  int64_t size2 = ort.GetTensorShapeElementCount(output_info2);
  ort.ReleaseTensorTypeAndShapeInfo(output_info2);

  for (int64_t i = 0; i < size2-1; i++) {
    out2[i] = 2.0f;
  }
}

void RunBeamSearchOnInternalSession(OrtKernelContext* context, OrtApi &api, Ort::CustomOpApi &ort, OrtSession *session) {
  std::cout<<"From beam search file"<<std::endl;
  std::vector<OrtValue*> inputs;
  std::vector<const char*> input_names{"input_ids", "position_ids", "attention_mask", "past_0", "past_1", "past_2", "past_3", "past_4", "past_5"};

  OrtMemoryInfo *ortmemoryinfo;
  // Must be freed explicitly
  api.CreateMemoryInfo("Cpu", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeCPU, &ortmemoryinfo);

  const OrtValue* input_ids_tensor = ort.KernelContext_GetInput(context, 0);
  const int* input_ids = ort.GetTensorData<int>(input_ids_tensor);
  
  OrtTensorTypeAndShapeInfo* input_ids_info = ort.GetTensorTypeAndShape(input_ids_tensor);
  std::vector<int64_t> tensor_shape = ort.GetTensorShape(input_ids_info);
  ort.ReleaseTensorTypeAndShapeInfo(input_ids_info);

  int64_t batch_size = tensor_shape[0];
  int64_t seq_len = tensor_shape[1];

  std::cout<<"Batch_size:"<<batch_size<<std::endl;
  std::cout<<"Seq_len:"<<seq_len<<std::endl;
  
  std::vector<int> input_ids_data;
  std::vector<int> position_ids_data;
  std::vector<int> attention_mask_data;
  for (int i = 0; i < batch_size; i++) {
    for (int j = 0; j < seq_len; j++) {
      input_ids_data.push_back(input_ids[i*seq_len+j]);
      position_ids_data.push_back(j);
      attention_mask_data.push_back(1);
    }
  }

  OrtValue *ort_input_ids;
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, input_ids_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &ort_input_ids);
  inputs.emplace_back(std::move(ort_input_ids));

  OrtValue *pos_ids;
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, position_ids_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &pos_ids);
  inputs.emplace_back(std::move(pos_ids));

  OrtValue *attn_mask;
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, attention_mask_data.data(), 4 * batch_size * seq_len, tensor_shape.data(), tensor_shape.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &attn_mask);
  inputs.emplace_back(std::move(attn_mask));

  int64_t past_seq_len = 0;
  int64_t num_heads = 16;
  int64_t head_size = 64;

  std::vector<int64_t> past_dims{2, batch_size, num_heads, past_seq_len, head_size};
  std::vector<float> past_data;
  
  for (int i = 0; i < 6; i++) {
    OrtValue *ort_past_data;
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, past_data.data(), 0, past_dims.data(), past_dims.size(),
        ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_past_data);

    inputs.emplace_back(std::move(ort_past_data));
  }

  std::vector<const char*> output_names{"logits", "present_0", "present_1", "present_2", "present_3", "present_4", "present_5"};
  std::vector<OrtValue*> outputs;
  std::vector<float> logits_data(batch_size * seq_len * 50263, 0);
  std::vector<int64_t> logits_dims{batch_size, seq_len, 50263};
  
  OrtValue* ortlogits;
  api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, logits_data.data(), 4*batch_size*seq_len*50263, logits_dims.data(),
      logits_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ortlogits);
  outputs.emplace_back(std::move(ortlogits));

  std::vector<float> present_0_data(2 * batch_size * num_heads * (seq_len+past_seq_len) * head_size);
  std::vector<int64_t> present_dims{2, batch_size, num_heads, seq_len+past_seq_len, head_size};
  for (int i = 0; i < 6; i++) {
    OrtValue* ort_present;
    // ort.present_1, ort.present_2, ort.present_3, ort.present_4, ort.present_5;
    api.CreateTensorWithDataAsOrtValue(ortmemoryinfo, present_0_data.data(), 4*2*batch_size*num_heads*(seq_len+past_seq_len)*head_size, present_dims.data(),
        present_dims.size(), ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_present);
    outputs.emplace_back(ort_present);
  }
  api.Run(session, nullptr, input_names.data(), inputs.data(), 9, output_names.data(), 7, outputs.data());

  std::cout<<"Printing logits"<<std::endl;
  int* logits = ort.GetTensorMutableData<int>(outputs[0]);
  for (int i = 0; i < batch_size; i++) {
    std::cout<<"batch:"<<i<<std::endl;
    for (int j = 0; j < seq_len; j++) {
      std::cout<<logits[i*seq_len + j]<<",";
    }
    std::cout<<std::endl;
  }

  SetBeamSearchOutputToZero(context, ort, batch_size, seq_len);
}