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


import argparse
import signal
import time
from enum import IntEnum

import argcomplete
import onnx
from app_helpers import (
    compute_module_args,
    create_input_buffer,
    init_compute_module,
    parse_output_buffer,
    print_output_values,
    run_and_get_results,
    sigint_handler,
)
from tqdm import tqdm


class Commands(IntEnum):
    CalculateNoCommand = 0
    RunInference = 1


def parse_arguments(
    explicit_args: list[str] | None = None,
):
    parser = argparse.ArgumentParser(
        description="Host application for the Signaloid C0 compute modules "
        "ONNX MLP model demo"
    )

    compute_module_args(parser=parser)

    parser.add_argument(
        "--model",
        type=str,
        help="Path of the ONNX model loaded onto the Signaloid C0 compute "
        "module. Default: ../signaloid-soc-application/model_a.onnx",
        default="../signaloid-soc-application/model_a.onnx",
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        help="Input data",
        required=True,
    )

    parser.add_argument(
        "--skip-printing-results",
        action="store_true",
        help="Skip printing the resulting Ux-Strings. "
        "Useful when benchmarking.",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--skip-plotting-results",
        action="store_true",
        help="Skip plotting the resulting Ux-Strings. "
        "Useful when benchmarking.",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--benchmark",
        default=False,
        action="store_true",
        help="Enable benchmarking",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="Benchmarking iterations. Default: 20",
    )

    argcomplete.autocomplete(parser)
    args = parser.parse_args(explicit_args)
    return args


def read_onnx_model(
    args: argparse.Namespace,
) -> tuple[int, int]:
    input_size: int = 1
    output_size: int = 1

    model: onnx.ModelProto = onnx.load(args.model)

    for dim in model.graph.input[0].type.tensor_type.shape.dim:
        input_size *= dim.dim_value

    for dim in model.graph.output[0].type.tensor_type.shape.dim:
        output_size *= dim.dim_value

    return (input_size, output_size)


def main(explicit_args: list[str] | None = None):
    signal.signal(signal.SIGINT, sigint_handler)

    args = parse_arguments(explicit_args)

    compute_module = init_compute_module(
        device_path=args.device_path,
        variant=args.variant,
        reset_on_launch=args.reset_on_launch,
    )

    input_count, output_count = read_onnx_model(args=args)

    command_value = Commands.RunInference

    input_buffer = create_input_buffer(
        values=args.input,
        buffer_size=compute_module.INPUT_BUFFER_SIZE_BYTES,
    )

    if args.benchmark:
        iterations = args.iterations
    else:
        iterations = 1

    totalDuration: float = 0
    result_buffer = bytes()
    for _ in tqdm(range(iterations), disable=not args.benchmark):
        startTime = time.perf_counter()

        # Run the calculation and get the results
        result_buffer = run_and_get_results(
            compute_module=compute_module,
            command_value=command_value,
            input_buffer=input_buffer,
            stop_on_exit=args.stop_on_exit,
            verbose=not args.benchmark,
        )

        endTime = time.perf_counter()
        iterationTime = endTime - startTime
        totalDuration += iterationTime

    if args.benchmark:
        meanTime = totalDuration / iterations
        print(
            f"Mean execution time over {iterations} ",
            f"iterations: {meanTime:.6f} seconds",
        )

    output_values = parse_output_buffer(
        buffer=result_buffer,
        expected_output_count=output_count,
    )
    print_output_values(
        values=output_values,
        skip_printing=args.skip_printing_results,
        skip_plotting=args.skip_plotting_results,
    )


if __name__ == "__main__":
    main()
