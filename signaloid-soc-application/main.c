/*
 *	Copyright (c) 2026, Signaloid.
 *
 *	Permission is hereby granted, free of charge, to any person obtaining a copy
 *	of this software and associated documentation files (the "Software"), to deal
 *	in the Software without restriction, including without limitation the rights
 *	to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *	copies of the Software, and to permit persons to whom the Software is
 *	furnished to do so, subject to the following conditions:
 *
 *	The above copyright notice and this permission notice shall be included in all
 *	copies or substantial portions of the Software.
 *
 *	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *	IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *	FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *	AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *	LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *	OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *	SOFTWARE.
 */


#include <stdint.h>
#include <uxhw.h>
#include <OnnxMlirRuntime.h>
#include "C0HAL.h"
#include "onnx_model_a_constants.h"


typedef enum
{
	kCalculateNoCommand = 0,
	kRunInference       = 1,
} SignaloidSoCCommand;


struct ModelData
{
	float           xData[k_onnx_model_a_input_tensors][k_onnx_model_a_input_size];
	OMTensor *      tensors[k_onnx_model_a_input_tensors];
	OMTensorList *  inputList;
	OMTensorList *  outputList;
	OMTensor *      y;
	float *         outputPtr;
};


/*
 * Helper functions
 */

SignaloidSoCCommand
waitForCommand(void)
{
	SignaloidSoCCommand command = kCalculateNoCommand;

	/*
	 *	Set status to "waitingForCommand"
	 */
	C0HALSetStatusRegister(kSignaloidSoCStatusWaitingForCommand);

	/*
	 *	Block until command is issued
	 */
	while (command == kCalculateNoCommand)
	{
		command = C0HALGetCommandRegister();
	}

	return command;
}

void
waitForIdle(void)
{
	/*
	 *	Block until command is cleared
	 */
	while (C0HALGetCommandRegister() != kCalculateNoCommand) {}
}

uint32_t
loadFloatOrUxBinary(volatile uint8_t * buffer, float * value)
{
	union
	{
		float       float32;
		uint32_t    uint32;
		uint8_t     byte[4];
	}
	u;

	u.byte[0]   = buffer[0];
	u.byte[1]   = buffer[1];
	u.byte[2]   = buffer[2];
	u.byte[3]   = buffer[3];
	buffer      += sizeof(uint32_t);

	uint32_t length = u.uint32;
	if (length == 4)
	{
		u.byte[0]   = buffer[0];
		u.byte[1]   = buffer[1];
		u.byte[2]   = buffer[2];
		u.byte[3]   = buffer[3];

		*value = UxHwFloatGaussDist(u.float32, 0.1);
	}
	else
	{
		*value = UxHwFloatByteArrayToDistribution((uint8_t *) buffer, length);
	}

	return length + sizeof(uint32_t);
}

uint32_t
writeUxBinary(float result, volatile uint8_t * buffer, uint32_t bufferBytes)
{
	if (bufferBytes <= sizeof(uint32_t))
	{
		return 0;
	}

	ssize_t resultSize = UxHwFloatDistributionToByteArray(
		result,
		((uint8_t *) buffer) + sizeof(uint32_t),
		bufferBytes - sizeof(uint32_t)
	);

	if (resultSize < 0)
	{
		*((uint32_t *) buffer) = 0;
		return sizeof(uint32_t);
	}

	*((uint32_t *) buffer) = (uint32_t) resultSize;

	return sizeof(uint32_t) + (uint32_t) resultSize;
}

uint32_t
loadModelInputTensors(struct ModelData * mdPtr)
{
	const int64_t   shape[]     = { 1, k_onnx_model_a_input_size };
	const int64_t   rank        = 2;
	uint32_t        inputCount  = 0;

	if (
		kC0HALInputBufferUint32[0] !=
		(k_onnx_model_a_input_tensors * k_onnx_model_a_input_size)
	)
	{
		return 0;
	}

	volatile uint8_t * buffer = kC0HALInputBufferUint8 + sizeof(uint32_t);
	for (uint32_t tensorIndex = 0; tensorIndex < k_onnx_model_a_input_tensors; tensorIndex++)
	{
		for (uint32_t valueIndex = 0; valueIndex < k_onnx_model_a_input_size; valueIndex++)
		{
			buffer += loadFloatOrUxBinary(
				buffer,
				&mdPtr->xData[tensorIndex][valueIndex]
			);
		}

		mdPtr->tensors[tensorIndex] = omTensorCreate(
			mdPtr->xData[tensorIndex],
			shape,
			rank,
			ONNX_TYPE_FLOAT
		);

		if (mdPtr->tensors[tensorIndex] == NULL)
		{
			return 0;
		}

		inputCount++;
	}

	return inputCount;
}

void
writeModelOutputTensor(struct ModelData * mdPtr)
{
	kC0HALOutputBufferUint32[0] = k_onnx_model_a_output_size;
	volatile uint8_t *  buffer      = kC0HALOutputBufferUint8 + sizeof(uint32_t);
	const uint8_t *     bufferEnd   = ((const uint8_t *) kC0HALOutputBufferUint8) + kC0HALOutputBufferUint8Length;
	for (uint32_t i = 0; i < k_onnx_model_a_output_size; i++)
	{
		buffer += writeUxBinary(
			mdPtr->outputPtr[i],
			buffer,
			bufferEnd - buffer
		);
	}
}

/*
 * Application Logic
 */

void
cleanup(struct ModelData * md)
{
	if (md->inputList != NULL)
	{
		omTensorListDestroy(md->inputList);
		md->inputList = NULL;
	}

	if (md->outputList != NULL)
	{
		omTensorListDestroy(md->outputList);
		md->outputList = NULL;
	}

	for (uint8_t i = 0; i < k_onnx_model_a_input_tensors; i++)
	{
		md->tensors[i] = NULL;
	}

	md->y           = NULL;
	md->outputPtr   = NULL;
}

void
runModelInference(void)
{
	struct ModelData md = {
		.inputList  = NULL,
		.outputList = NULL,
		.y          = NULL,
		.outputPtr  = NULL,
	};

	if (loadModelInputTensors(&md) != k_onnx_model_a_input_tensors)
	{
		cleanup(&md);
		C0HALSetStatusRegister(kSignaloidSoCStatusInvalidCommand);

		return;
	}

	/*
	 *	Construct a list of omts as input.
	 */
	md.inputList = omTensorListCreate(md.tensors, k_onnx_model_a_input_tensors);
	if (md.inputList == NULL)
	{
		cleanup(&md);
		C0HALSetStatusRegister(kSignaloidSoCStatusInvalidCommand);

		return;
	}

	/*
	 *	Call the compiled onnx model function.
	 */
	md.outputList = run_main_graph_model_a(md.inputList);
	if (md.outputList == NULL)
	{
		cleanup(&md);
		C0HALSetStatusRegister(kSignaloidSoCStatusInvalidCommand);

		return;
	}

	/*
	 *	Get the first omt as output.
	 */
	md.y            = omTensorListGetOmtByIndex(md.outputList, 0);
	md.outputPtr    = (float *) omTensorGetDataPtr(md.y);

	writeModelOutputTensor(&md);

	cleanup(&md);
}

void
handleRunInference(void)
{
	/*
	 *	Set status to inform host that calculation will start
	 */
	C0HALSetStatusRegister(kSignaloidSoCStatusCalculating);

	/*
	 *	Turn on status LED
	 */
	C0HALSetLed(true);

	runModelInference();

	/*
	 *	Turn off status LED
	 */
	C0HALSetLed(false);

	/*
	 *	Set status
	 */
	C0HALSetStatusRegister(kSignaloidSoCStatusDone);
}

void
handleCommand(SignaloidSoCCommand command)
{
	switch (command)
	{
		case kRunInference:
			handleRunInference();
			break;

		default:
			C0HALSetStatusRegister(kSignaloidSoCStatusInvalidCommand);
			break;
	}
}

void
setup(void)
{
	C0HALSetLed(false);
	C0HALSetStatusRegister(kSignaloidSoCStatusWaitingForCommand);
}

void
loop(void)
{
	SignaloidSoCCommand command = waitForCommand();
	handleCommand(command);
	waitForIdle();
}

int
main(void)
{
	setup();
	while (1)
	{
		loop();
	}
}
