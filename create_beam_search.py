import argparse
import json
from typing import List, Optional
import onnx
from onnx import helper, TensorProto
import os
import shutil

'''
Usage:

Use this to convert a decoder model to run with custombeamsearchop
https://github.com/viboga/ORTCustomBeamsearchOpLibrary

Example:
python create_beam_search.py --input_model_path D:\ai\onnx\gpt2.onnx --vocab_size 50263 --output_model_path D:\ai\onnx\custombsop\gpt2_beamsearch.onnx
'''

custom_op_type = "CustomBeamsearchOp"
custom_domain = "test.beamsearchop"

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-i',
                        '--input_model_path',
                        required=True,
                        type=str,
                        help='Model name')

    parser.add_argument('-o',
                        '--output_model_path',
                        required=True,
                        type=str,
                        help='Path where the model has to be saved')

    parser.add_argument("--vocab_size",
                        type=int,
                        required=True,
                        help="Vocab_size of the underlying model used to decide the shape of vocab mask")

    parser.add_argument("--eos_token_id",
                        type=int,
                        required=True,
                        help="end of sequence token id")

    parser.add_argument("--pad_token_id",
                        type=int,
                        required=True,
                        help="end of sequence token id")

    parser.add_argument("--no_repeat_ngram_size",
                        type=int,
                        required=False,
                        default=0,
                        help="no repeat n gram size")

    parser.add_argument("--num_layers",
                        type=int,
                        required=False,
                        default=6,
                        help="expected num of layers in the inner graph")

    parser.add_argument("--num_return_sequences",
                        type=int,
                        required=False,
                        default=1,
                        help="expected num of return sequences")

    parser.add_argument("--early_stopping",
                        type=int,
                        required=False,
                        default=0,
                        help="no repeat n gram size")

    args = parser.parse_args(argv)

    return args

def convert_to_beam_search(model: onnx.ModelProto, args: argparse.Namespace):
    inputs = ["input_ids", "max_length", "vocab_mask", "prefix_vocab_mask"]
    outputs = ["sequences", "sequences_scores"]

    node = onnx.helper.make_node(
        custom_op_type,
        inputs=inputs,
        outputs=outputs,
        name=f"CustomBeamSearch_0",
        domain = custom_domain
    )

    output_model_path = os.path.normpath(args.output_model_path)
    output_model_folder = os.path.dirname(output_model_path)
    if not os.path.exists(os.path.join(output_model_folder, "model")):
        os.makedirs(os.path.join(output_model_folder, "model"))
    internal_model_path =  os.path.join(output_model_folder, "model", "decoder.onnx")
    beamsearch_config_path = os.path.join(output_model_folder, "model", "config.json")
    config = {
        "vocab_size" : args.vocab_size,
        "num_heads": 16,
        "head_size": 64,
        "num_layers": args.num_layers,
        "length_penalty": 1.0,
        "repetetion_penalty": 1.0,
        "num_beams" : 1,
        "min_length" : 1,
        "max_words": 6,
        "num_return_sequences" : args.num_return_sequences,
        "first_past_input_idx" : 3,
        "first_present_output_idx": 1
    }
    json.dump(config, open(beamsearch_config_path, 'w'))
    shutil.copyfile(args.input_model_path, internal_model_path)
    
    node.attribute.extend(
        [
            onnx.helper.make_attribute("eos_token_id", args.eos_token_id),
            onnx.helper.make_attribute("pad_token_id", args.pad_token_id),
            onnx.helper.make_attribute("no_repeat_ngram_size", args.no_repeat_ngram_size),
            onnx.helper.make_attribute("early_stopping", 1 if args.early_stopping else 0),
            onnx.helper.make_attribute("model_path", internal_model_path)
        ]
    )

    input_ids = onnx.helper.make_tensor_value_info('input_ids', TensorProto.INT32, ['batch_size', 'sequence_length'])
    max_length = onnx.helper.make_tensor_value_info("max_length", TensorProto.INT32, [1])
    vocab_mask = onnx.helper.make_tensor_value_info('vocab_mask', TensorProto.INT32, [args.vocab_size])
    prefix_vocab_mask = onnx.helper.make_tensor_value_info('prefix_vocab_mask', TensorProto.INT32, ['batch_size', args.vocab_size])
    graph_inputs = [input_ids, max_length, vocab_mask, prefix_vocab_mask]

    sequences = onnx.helper.make_tensor_value_info("sequences", TensorProto.INT32, ["batch_size", args.num_return_sequences, "max_length"])
    sequences_scores = onnx.helper.make_tensor_value_info("sequences_scores", TensorProto.FLOAT, ["batch_size", args.num_return_sequences])
    graph_outputs = [sequences, sequences_scores]

    new_graph = onnx.helper.make_graph([node], f"decoder beam search", graph_inputs, graph_outputs)
    new_model = onnx.helper.make_model(new_graph, producer_name="onnxruntime.beamsearch.customop",opset_imports=model.opset_import)

    onnx.save(new_model, args.output_model_path)

def verify_gpt2_decoder_graph(graph: onnx.GraphProto):
    input_count = len(graph.input)
    layer_count = input_count - 3
    assert layer_count >= 1

    expected_inputs = ["input_ids", "position_ids", "attention_mask"] + [f"past_{i}" for i in range(layer_count)]
    if len(graph.input) != len(expected_inputs):
        raise ValueError(f"Number of inputs expected to be {len(expected_inputs)}. Got {len(graph.input)}")
    
    for i, expected_input in enumerate(expected_inputs):
        if graph.input[i].name != expected_input:
            raise ValueError(f"Input {i} is expected to be {expected_input}. Got {graph.input[i].name}")

        expected_type = TensorProto.INT32
        if i >= 3:
            expected_type = TensorProto.FLOAT

        input_type = graph.input[i].type.tensor_type.elem_type
        if input_type != expected_type:
            if i<3:
                print(f"Input {i} is expected to be int32, but it is {input_type}, converting the input to int32")
                # TODO Directly setting it to INT32, verify there is a cast node exists below the inputs
                graph.input[i].type.tensor_type.elem_type = expected_type
            else:
                raise ValueError(f"Input {i} is expected to have onnx data type {expected_type}. Got {input_type}")

    expected_outputs = ["logits"] + [f"present_{i}" for i in range(layer_count)]
    if len(graph.output) != len(expected_outputs):
        raise ValueError(f"Number of outputs expected to be {len(expected_outputs)}. Got {len(graph.output)}")

    expected_type = TensorProto.FLOAT
    for i, expected_output in enumerate(expected_outputs):
        if graph.output[i].name != expected_output:
            raise ValueError(f"Output {i} is expected to be {expected_output}. Got {graph.output[i].name}")

        output_type = graph.output[i].type.tensor_type.elem_type
        if output_type != expected_type:
            raise ValueError(f"Output {i} is expected to have onnx data type {expected_type}. Got {output_type}")

    return graph

def make_custombeamsearchop(args):
    if not os.path.exists(args.input_model_path):
        raise Exception(f"Input path doesn't exist")

    #Note: external data is not supported now, like loading weights seperately
    model = onnx.load_model(args.input_model_path)
    verify_gpt2_decoder_graph(model.graph)

    convert_to_beam_search(model, args)

def main(argv: Optional[List[str]] = None):
    args = parse_arguments(argv)
    make_custombeamsearchop(args)

if __name__ == '__main__':
    """
    Usage: On powershell:
    python create_beam_search.py -i D:\ai\onnx\gpt2.onnx `
                                 --vocab_size 50263  `
                                 -o D:\ai\onnx\custombsop\gpt2_beamsearch.onnx `
                                 --eos_token_id 50256 `
                                 --pad_token_id 50262
    """
    main()