#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <iostream>
#include <iomanip>

#include "onnxruntime_cxx_api.h"

#include <assert.h>
#include "run_custom_bs.h"

#define TEST_QUERIES 1
#define CUSTOM_OP 1

#include <fstream>
#include <sstream>

using namespace std;

void RunCustomOpBeamsearchTestModel(std::wstring &wide_string, const char *custom_op_library_filename, const char *test_data_file, const char *result_file)
{
  int num_words = 6;
  int batch_size = 1;
  int num_return_sequences = 1;

  Ort::Env ort_env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test custom op");

  Ort::SessionOptions session_options = Ort::SessionOptions();

  void *handle = nullptr;
  Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions *)session_options, custom_op_library_filename, &handle);
  Ort::Session session(ort_env, wide_string.c_str(), session_options);

  Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<Ort::Value> ort_inputs;
  std::vector<const char *> input_names;

  // input_ids
  input_names.emplace_back("input_ids");
  input_names.emplace_back("max_length");
  input_names.emplace_back("vocab_mask");
  input_names.emplace_back("prefix_vocab_mask");

  std::vector<const char *> output_names;
  output_names.emplace_back("sequences");
  output_names.emplace_back("sequences_scores");

  int query_counter = 0;
  int max_length = 0;

  fstream fileStream;
  fileStream.open(test_data_file);

  fstream fileOutStream;
  fileOutStream.open(result_file, fstream::out);

  while (1)
  {
    std::string line;
    std::getline(fileStream, line);

    // Total queries in the file are hundred
    if (line.length() == 0)
    {
      break;
    }
    std::stringstream lineStream(line);
    int input_count;
    lineStream >> input_count;

    query_counter++;

    std::vector<int> input_ids;
    int input_id;
    for (int ii = 0; ii < input_count; ii++)
    {
      lineStream >> input_id;
      input_ids.push_back(input_id);
    }

    std::vector<int64_t> input_ids_shape{batch_size, input_count};
    ort_inputs.emplace_back(std::move(Ort::Value::CreateTensor<int>(memory_info,
                                                                    input_ids.data(), input_ids.size(),
                                                                    input_ids_shape.data(), input_ids_shape.size())));

    max_length = input_count + num_words;
    std::vector<int> ml{max_length};
    std::vector<int64_t> ml_shape{1};
    ort_inputs.emplace_back(std::move(Ort::Value::CreateTensor<int>(memory_info,
                                                                    ml.data(), ml.size(),
                                                                    ml_shape.data(), ml_shape.size())));

    std::vector<int> vm_data(50263, 1);
    std::vector<int64_t> vm_shape{50263};
    ort_inputs.emplace_back(std::move(Ort::Value::CreateTensor<int>(memory_info,
                                                                    vm_data.data(), vm_data.size(),
                                                                    vm_shape.data(), vm_shape.size())));

    std::getline(fileStream, line);
    std::stringstream lineStream_pvm(line);

    int pvm_count;
    lineStream_pvm >> pvm_count;

    std::vector<int> pvm_data(batch_size * 50263, 0);
    int pvm_id;
    for (int i = 0; i < pvm_count; i++)
    {
      lineStream_pvm >> pvm_id;
      pvm_data[pvm_id] = 1;
    }
    std::vector<int64_t> pvm_shape{batch_size, 50263};
    ort_inputs.emplace_back(std::move(Ort::Value::CreateTensor<int>(memory_info,
                                                                    pvm_data.data(), pvm_data.size(),
                                                                    pvm_shape.data(), pvm_shape.size())));
    auto ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names.data(), ort_inputs.data(),
                                   ort_inputs.size(), output_names.data(), output_names.size());

    int *gen_sequences = ort_outputs[0].GetTensorMutableData<int>();

    for (size_t i = 0; i < batch_size; i++)
    {
      for (size_t j = 0; j < num_return_sequences; j++)
      {
        for (size_t k = input_count; k < input_count + num_words; k++)
        {
          fileOutStream << gen_sequences[i * num_return_sequences + j * max_length + k] << " ";
        }
      }
      fileOutStream << "\n";
    }

    ort_inputs.clear();
  }

  fileStream.close();
  fileOutStream.close();

  std::cout << "THE END" << std::endl;
}

int main(int argc, char **argv)
{
  int a;
  std::cout << "Enter any number to start the test:";
  std::cin >> a;

  if (argc != 5)
  {
    std::cout << "Usage: cc_app.exe <onnx_model.onnx> <custom beam search op dll path> <test_data_file> <result_file>" << std::endl;
    return -1;
  }

  std::string str = argv[1];
  std::wstring wide_string = std::wstring(str.begin(), str.end());

  RunCustomOpBeamsearchTestModel(wide_string, std::string(argv[2]).c_str(), std::string(argv[3]).c_str(), std::string(argv[4]).c_str());
  return 0;
}