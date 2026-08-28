#!/usr/bin/env python -u
# PYTHON_ARGCOMPLETE_OK

#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.


"""
This script generates an MLP-type ONNX model parametrically.

It uses the list of layer sizes to generate the needed layers.
The first number in the list is the input size.
The next N numbers dictate the width of the N Fully-Connected layers.
Each Fully-Connected layer is followed by a ReLU activation function.
The last number in the list dictates the width of the SoftMax output layer.

This script saves the generated ONNX model at the given model path, generates
the needed C header file for the signaloid SoC application at the given header
path, and runs the model with all input values normalized to 1 / input_size to
print the expected model output.
"""


import argparse

import argcomplete
import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


DEFAULT_MODEL_FILENAME = "model_a.onnx"
DEFAULT_HEADER_FILENAME = "onnx_model_a_constants.h"

# The size of the distributional representation.
# Used to estimate the needed memory.
DEFAULT_REPRESENTATION_SIZE = 8

# The width of each Fully-Connected Layer.
# First number is the input width.
# Last number is the output width.
DEFAULT_LAYER_SIZES = [10, 20, 30, 2]

np.random.seed(20260809)


def gen_model(
    layer_sizes: list[int],
    representation_size: int = DEFAULT_REPRESENTATION_SIZE,
) -> onnx.ModelProto:
    node_list: list[onnx.NodeProto] = []
    initializer: list[onnx.TensorProto] = []
    total_params: int = 0
    total_params_bytes: int = 0
    for i in range(len(layer_sizes) - 1):
        # Do NOT use zeroed out weights and biases since they are going to be
        # compressed when compiled with ONNX MLIR, and the binary will not be
        # realistic
        weights = (
            np.random.random((layer_sizes[i], layer_sizes[i + 1])).astype(
                np.float32
            )
            * 0.5
            / layer_sizes[i + 1]
        )
        biases = (
            np.random.random((1, layer_sizes[i + 1])).astype(np.float32)
            * 0.5
            / layer_sizes[i + 1]
        )

        initializer.append(
            onnx.numpy_helper.from_array(weights, name=f"weights{i}")
        )
        initializer.append(
            onnx.numpy_helper.from_array(biases, name=f"biases{i}")
        )

        total_params += weights.size
        total_params += biases.size
        total_params_bytes += weights.nbytes
        total_params_bytes += biases.nbytes

        node_list.append(
            onnx.helper.make_node(
                op_type="MatMul",
                inputs=[f"X{i}", f"weights{i}"],
                outputs=[f"MatMul_output{i}"],
            )
        )
        node_list.append(
            onnx.helper.make_node(
                op_type="Add",
                inputs=[f"MatMul_output{i}", f"biases{i}"],
                outputs=[f"Add_output{i}"],
            )
        )
        node_list.append(
            onnx.helper.make_node(
                op_type="Relu",
                inputs=[f"Add_output{i}"],
                outputs=[f"X{i+1}"],
            )
        )

    node_list.append(
        onnx.helper.make_node(
            op_type="Softmax",
            inputs=[f"X{len(layer_sizes) - 1}"],
            outputs=["Y"],
        )
    )

    X = onnx.helper.make_tensor_value_info(
        name="X0",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[1, layer_sizes[0]],
    )
    Y = onnx.helper.make_tensor_value_info(
        name="Y",
        elem_type=onnx.TensorProto.FLOAT,
        shape=[1, layer_sizes[-1]],
    )

    graph = onnx.helper.make_graph(
        nodes=node_list,
        name="lr",
        inputs=[X],
        outputs=[Y],
        initializer=initializer,
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)

    print(f"- Model Depth : {len(layer_sizes):>7}")
    print(f"- Total Params: {total_params:>7} ({total_params_bytes:>7} B)")

    distribution_bytes = representation_size * 8

    total_inputs = layer_sizes[0]
    total_inputs_bytes = total_inputs * distribution_bytes
    print(f"- Total Inputs: {total_inputs:>7} ({total_inputs_bytes:>7} B)")

    total_nodes = sum(layer_sizes) + layer_sizes[-1]
    total_nodes_bytes = total_nodes * distribution_bytes
    print(f"- Total Nodes : {total_nodes:>7} ({total_nodes_bytes:>7} B)")

    total_memory = total_params_bytes + sum(layer_sizes) * distribution_bytes
    print(f"- Total Memory: {'':>7}  {total_memory:>7} B")

    return onnx_model


def run_model(
    model: onnx.ModelProto,
    layer_sizes: list[int],
):
    input_data = (
        np.ones(layer_sizes[0]).astype(np.float32).reshape(1, layer_sizes[0])
        / layer_sizes[0]
    )
    runner = ReferenceEvaluator(model)
    y = runner.run(output_names=None, feed_inputs={"X0": input_data})
    print("- Model Input :", input_data[0])
    print("- Model Output:", y[0][0])
    return y


def gen_header_file(
    model_name: str,
    layer_sizes: list[int],
    header_path: str,
):
    text = ""
    text += "/*\n"
    text += " * This file is autogenerated from gen_mlp_onnx.py.\n"
    text += " * DO NOT EDIT.\n"
    text += " */\n"
    text += "\n"
    text += "#pragma once\n"
    text += "\n"
    text += f"#define k_onnx_{model_name}_input_tensors 1\n"
    text += f"#define k_onnx_{model_name}_input_size {layer_sizes[0]}\n"
    text += f"#define k_onnx_{model_name}_output_size {layer_sizes[-1]}\n"
    text += "\n"
    text += "\n"
    text += "OMTensorList *\n"
    text += f"run_main_graph_{model_name}(OMTensorList *);"

    with open(header_path, "w") as f:
        f.write(text)


def parse_arguments(
    explicit_args: list[str] | None = None,
):
    parser = argparse.ArgumentParser(
        description="Generate MLP-type ONNX models parametrically"
    )

    parser.add_argument(
        "--layer-sizes",
        nargs="+",
        type=int,
        help=f"Layer sizes. Defaults: {DEFAULT_LAYER_SIZES}",
        default=DEFAULT_LAYER_SIZES,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        help=f"Path of the generated ONNX model. "
        f"Default: {DEFAULT_MODEL_FILENAME}",
        default=DEFAULT_MODEL_FILENAME,
    )

    parser.add_argument(
        "--header-path",
        type=str,
        help=f"Path of the generated C header file. "
        f"Default: {DEFAULT_HEADER_FILENAME}",
        default=DEFAULT_HEADER_FILENAME,
    )

    parser.add_argument(
        "--representation-size",
        type=int,
        help=f"The SoC application representation size. "
        f"Used for estimating the memory requirements. "
        f"Default: {DEFAULT_REPRESENTATION_SIZE}",
        default=DEFAULT_REPRESENTATION_SIZE,
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args(explicit_args)
    return args


def main(
    explicit_args: list[str] | None = None,
):
    args = parse_arguments(explicit_args=explicit_args)

    onnx_model = gen_model(
        args.layer_sizes,
        representation_size=args.representation_size,
    )
    onnx.save(onnx_model, args.model_path)
    gen_header_file(
        args.model_path.replace(".onnx", ""),
        args.layer_sizes,
        args.header_path,
    )

    run_model(model=onnx_model, layer_sizes=args.layer_sizes)


if __name__ == "__main__":
    main()
