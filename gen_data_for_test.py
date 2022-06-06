from asyncore import write
import numpy as np
from torch import tensor
import onnxruntime as ort
import onnx
import os
import pickle

from onnx import numpy_helper

model_folder = os.path.dirname(os.path.abspath(__file__))
path = model_folder + "model.onnx"


def read_with_pickle(file_path):
    with open(file_path, 'rb') as fp:
        itemlist = pickle.load(fp)
    print(itemlist)


def generate_mask_with_1s(dim):
    attn_mask = np.array(np.ones((1,dim), dtype=np.int64))
    tensor_1 = numpy_helper.from_array(attn_mask, 'input_mask')

    with open(os.path.join(".\\", 'input_{}.pb'.format(2)), 'wb') as f:
        f.write(tensor_1.SerializeToString())


def generate_testdata(model_path):
    test_cycle = int(np.random.rand() * 1000)
    input_1 = np.array(np.random.randn(1, 3, 16).astype(np.float32))
    input_2 = np.array(np.random.randn(1, 3, 16).astype(np.float32))
    attn_mask = np.array(np.random.randn(1, 3).astype(np.int64))

    tensor_1 = numpy_helper.from_array(input_1, 'input_1')
    tensor_2 = numpy_helper.from_array(input_2, 'input_2')
    tensor_3 = numpy_helper.from_array(attn_mask, 'input_mask')

    print("tensors created")

    with open(os.path.join(model_folder, "test_data_set_0", 'input_{}.pb'.format(0)), 'wb') as f:
        f.write(tensor_1.SerializeToString())
    with open(os.path.join(model_folder, "test_data_set_0", 'input_{}.pb'.format(1)), 'wb') as f:
        f.write(tensor_2.SerializeToString())
    with open(os.path.join(model_folder, "test_data_set_0", 'input_{}.pb'.format(2)), 'wb') as f:
        f.write(tensor_3.SerializeToString())

    print("Written with sucess")

def write_to_file(model_folder, tensor_name : str,
                  dim1 : int, dim2 : int, tensor_input_number : int,
                  np_type : np.integer, random:bool, val: int = 0):
    if random:
        if dim1 != 0 and dim2 != 0:
            input = np.array(np.random.randn(dim1, dim2)).astype(np_type)
        else:
            input = np.array(np.random.randn(dim1)).astype(np_type)
    else:
        if dim1 != 0 and dim2 != 0:
            input = np.array(np.full((dim1, dim2), val)).astype(np_type)
        else:
            input = np.array(np.full(dim1, val)).astype(np_type)

    tensor = numpy_helper.from_array(input, tensor_name)
    with open(os.path.join(model_folder, "test_data_set_0", 'input_{}.pb'.format(tensor_input_number)), "wb") as f:
        f.write(tensor.SerializeToString())

def generate_test_data_beamsearch():
    model_folder = "D:\\ai\\AI frameworks Team\\tasks\\deepwrite\\DeepWritev1\\DeepWrite\\model\\debug_model\\"

    write_to_file(model_folder, 'input_ids', 1, 16, 0, np.int32, True)
    write_to_file(model_folder, 'max_length', 1, 0, 1, np.int32, False, 40)
    write_to_file(model_folder, 'min_length', 1, 0, 2, np.int32, False, 1)
    write_to_file(model_folder, 'num_beams', 1, 0, 3, np.int32, False, 2)
    write_to_file(model_folder, 'num_return_sequences', 1, 0, 4, np.int32, False, 2)
    write_to_file(model_folder, 'temperature', 1, 0, 5, np.float32, False, 1)
    write_to_file(model_folder, 'length_penalty', 1, 0, 6, np.float32, False, 1)
    write_to_file(model_folder, 'repetition_penalty', 1, 0, 7, np.float32, False, 1)
    write_to_file(model_folder, 'vocab_mask', 50263, 0, 8, np.int32, False, 1)
    write_to_file(model_folder, 'prefix_vocab_mask', 1, 50263, 9, np.int32, False, 1)
    write_to_file(model_folder, 'ecs_min_chars', 1, 0, 10, np.int32, False, 6)
    write_to_file(model_folder, 'ecs_log_prob_threshold', 1, 0, 11, np.float32, False, 1.344)
    write_to_file(model_folder, 'ecs_cost', 1, 0, 12, np.float32, False, 0)
    write_to_file(model_folder, 'vocab_ids2_len', 50263, 0, 13, np.int32, False, 3)
    write_to_file(model_folder, 'prefix_lens', 1, 0, 14, np.int32, False, 2)
    write_to_file(model_folder, 'prefix_uppercase', 1, 0, 15, np.bool8, False, 1)

#generate_testdata(path)
#generate_mask_with_1s(3)
generate_test_data_beamsearch()