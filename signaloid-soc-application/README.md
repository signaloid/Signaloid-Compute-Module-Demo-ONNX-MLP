# Signaloid SoC application

This directory holds the firmware that runs on the Signaloid SoC inside the
compute module.

The firmware is built in the Signaloid Cloud Developer Platform. Use the targets
in the top-level [Makefile](../Makefile) to build, download, and flash it.

## Files

| File                                                 | Purpose                                                                                                                    |
| ---------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| [main.c](main.c)                                     | Entry point. Polls the command register, dispatches to the selected command, and packs the results into the output buffer. |
| [config.mk](config.mk)                               | Build configuration. Selects the target device sources.                                                                    |
| [gen_mlp_onnx.py](gen_mlp_onnx.py)                   | The Python script that generates the ONNX model `model_a.onnx` file, linked during compilation of the SoC application.     |
| [model_a.onnx](model_a.onnx)                         | The ONNX model to use for the inference.                                                                                   |
| [onnx_model_a_constants.h](onnx_model_a_constants.h) | The ONNX model parameter definitions.                                                                                      |

## Configuration

`config.mk` sets the sources for the selected `DEVICE_TYPE` and the Signaloid
Compute Module Utilities path.

Add your own compiler flags through the `BUILD_FLAGS` variable, and add your
sources and include paths on the `SOURCES` and `INC` variables respectively of
`config.mk`.
