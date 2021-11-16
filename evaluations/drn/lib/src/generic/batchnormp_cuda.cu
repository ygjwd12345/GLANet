#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/batchnormp_cuda.cu"
#else

#define DeviceTensor3 THCDeviceTensor<real, 3>
#define DeviceTensor1 THCDeviceTensor<real, 1>

template <int Dim>
static THCDeviceTensor<real, Dim> devicetensor(THCState *state, THCTensor *t) {
  if (!t) {
    return THCDeviceTensor<real, Dim>();
  }

  int inDim = THCTensor_(nDimension)(state, t);
  if (inDim == Dim) {
    return toDeviceTensor<real, Dim>(state, t);
  }

  // View in which the last dimensions are collapsed or expanded as needed
  THAssert(THCTensor_(isContiguous)(state, t));
  int size[Dim];
  for (int i = 0; i < Dim || i < inDim; ++i) {
    if (i < Dim && i < inDim) {
      size[i] = t->size[i];
    } else if (i < Dim) {
      size[i] = 1;
    } else {
      size[Dim - 1] *= t->size[i];
    }
  }
  return THCDeviceTensor<real, Dim>(THCTensor_(data)(state, t), size);
}

extern "C" void THNN_(BatchNormalization_updateOutputhaha)(
  THCState *state, THCTensor *input_, THCTensor *output_,
  THCTensor *weight_, THCTensor *bias_, THCTensor *runningMean_,
  THCTensor *runningVar_, THCTensor *saveMean_, THCTensor *saveStd_,
  int train, double momentum, double eps);

extern "C" void THNN_(BatchNormalization_mean)(
  THCState *state, THCTensor *input_, THCTensor *saveMean_);

extern "C" void THNN_(BatchNormalization_var)(
  THCState *state, THCTensor *input_, THCTensor *saveMean_,
  THCTensor *saveVar_);


void THNN_(BatchNormalization_mean)(
  THCState *state, THCTensor *input_, THCTensor *saveMean_) {
  DeviceTensor3 input = devicetensor<3>(state, input_);
  DeviceTensor1 saveMean = devicetensor<1>(state, saveMean_);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);

  dim3 blocks(input.getSize(1));
  dim3 threads(getNumThreads(input.getSize(2)));
  BatchNormalizationMean_kernel<real, accreal, DeviceTensor1, DeviceTensor3> <<<blocks, threads, 0, s>>>(
  input, saveMean);
  THCudaCheck(cudaGetLastError());
}

void THNN_(BatchNormalization_var)(
  THCState *state, THCTensor *input_, THCTensor *saveMean_, THCTensor *saveVar_) {
  DeviceTensor3 input = devicetensor<3>(state, input_);
  DeviceTensor1 saveMean = devicetensor<1>(state, saveMean_);
  DeviceTensor1 saveVar = devicetensor<1>(state, saveVar_);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);

  dim3 blocks(input.getSize(1));
  dim3 threads(getNumThreads(input.getSize(2)));
  BatchNormalizationVar_kernel<real, accreal, DeviceTensor1, DeviceTensor3> <<<blocks, threads, 0, s>>>(
  input, saveMean, saveVar);
  THCudaCheck(cudaGetLastError());
}

void THNN_(BatchNormalization_updateOutputhaha)(
  THCState *state, THCTensor *input_, THCTensor *output_,
  THCTensor *weight_, THCTensor *bias_, THCTensor *runningMean_,
  THCTensor *runningVar_, THCTensor *saveMean_, THCTensor *saveStd_,
  int train, double momentum, double eps) {

  THCTensor_(resizeAs)(state, output_, input_);
  DeviceTensor3 input = devicetensor<3>(state, input_);
  DeviceTensor3 output = devicetensor<3>(state, output_);
  DeviceTensor1 weight = devicetensor<1>(state, weight_);
  DeviceTensor1 bias = devicetensor<1>(state, bias_);
  DeviceTensor1 runningMean = devicetensor<1>(state, runningMean_);
  DeviceTensor1 runningVar = devicetensor<1>(state, runningVar_);
  DeviceTensor1 saveMean = devicetensor<1>(state, saveMean_);
  DeviceTensor1 saveStd = devicetensor<1>(state, saveStd_);

  cudaStream_t s = THCState_getCurrentStream(state);
  cudaDeviceProp *prop = THCState_getCurrentDeviceProperties(state);

  if (!train) {
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    BatchNormalizationUpdateOutputInference_kernel<real, accreal, DeviceTensor1, DeviceTensor3> <<<blocks, threads, 0, s>>>(
      input, output, runningMean, runningVar, weight, bias, eps);
  } else {
    dim3 blocks(input.getSize(1));
    dim3 threads(getNumThreads(input.getSize(2)));
    BatchNormalizationUpdateOutput_kernelhaha<real, accreal, DeviceTensor1, DeviceTensor3> <<<blocks, threads, 0, s>>>(
      input, output, weight, bias, eps, momentum, runningMean, runningVar,
      saveMean, saveStd);
  }
  THCudaCheck(cudaGetLastError());
}

extern "C" void THNN_(BatchNormalization_backwardhaha)(
  THCState *state, THCTensor *input_, THCTensor *gradOutput_,
  THCTensor *gradOutputMean_, THCTensor *dotP,
  THCTensor *gradInput_, THCTensor *gradWeight_, THCTensor *gradBias_,
  THCTensor *weight_, THCTensor *runningMean_, THCTensor *runningVar_,
  THCTensor *saveMean_, THCTensor *saveStd_, int train, double scale, double eps);


extern "C" void THNN_(BatchNormalization_mean_grad)(
  THCState *state, THCTensor *input_, THCTensor *gradOutput_,
  THCTensor *runningMean_, THCTensor *saveMean_,
  THCTensor *gradOutputMean_, THCTensor *dotP_, int train);


void THNN_(BatchNormalization_mean_grad)(
  THCState *state, THCTensor *input_, THCTensor *gradOutput_,
  THCTensor *runningMean_, THCTensor *saveMean_,
  THCTensor *gradOutputMean_, THCTensor *dotP_, int train) {

  THCUNN_check_shape(state, input_, gradOutput_);
  DeviceTensor3 input = devicetensor<3>(state, input_);
  DeviceTensor3 gradOutput = devicetensor<3>(state, gradOutput_);
  DeviceTensor1 gradOutputMean = devicetensor<1>(state, gradOutputMean_);
  DeviceTensor1 dotP = devicetensor<1>(state, dotP_);

  DeviceTensor1 runningMean = devicetensor<1>(state, runningMean_);
  DeviceTensor1 saveMean = devicetensor<1>(state, saveMean_);

  cudaStream_t s = THCState_getCurrentStream(state);

  dim3 blocks(gradOutput.getSize(1));
  dim3 threads(getNumThreads(gradOutput.getSize(2)));
  BatchNormalizationMeanGrad_kernel<real,  accreal,  DeviceTensor1, DeviceTensor3> <<<blocks, threads, 0, s>>>(
    input, gradOutput, runningMean, saveMean, gradOutputMean, dotP, train);
  THCudaCheck(cudaGetLastError());
}


void THNN_(BatchNormalization_backwardhaha)(
  THCState *state, THCTensor *input_, THCTensor *gradOutput_,
  THCTensor *gradOutputMean_, THCTensor *dotP_,
  THCTensor *gradInput_, THCTensor *gradWeight_, THCTensor *gradBias_,
  THCTensor *weight_, THCTensor *runningMean_, THCTensor *runningVar_,
  THCTensor *saveMean_, THCTensor *saveStd_, int train, double scale, double eps) {

  THCUNN_check_shape(state, input_, gradOutput_);
  DeviceTensor3 input = devicetensor<3>(state, input_);
  DeviceTensor3 gradOutput = devicetensor<3>(state, gradOutput_);
  DeviceTensor1 gradOutputMean = devicetensor<1>(state, gradOutputMean_);
  DeviceTensor1 dotP = devicetensor<1>(state, dotP_);
  DeviceTensor3 gradInput = devicetensor<3>(state, gradInput_);
  DeviceTensor1 gradWeight = devicetensor<1>(state, gradWeight_);
  DeviceTensor1 gradBias = devicetensor<1>(state, gradBias_);
  DeviceTensor1 weight = devicetensor<1>(state, weight_);
  DeviceTensor1 runningMean = devicetensor<1>(state, runningMean_);
  DeviceTensor1 runningVar = devicetensor<1>(state, runningVar_);
  DeviceTensor1 saveMean = devicetensor<1>(state, saveMean_);
  DeviceTensor1 saveStd = devicetensor<1>(state, saveStd_);

  cudaStream_t s = THCState_getCurrentStream(state);

  dim3 blocks(gradOutput.getSize(1));
  dim3 threads(getNumThreads(gradOutput.getSize(2)));
  BatchNormalizationBackward_kernel<real,  accreal,  DeviceTensor1, DeviceTensor3> <<<blocks, threads, 0, s>>>(
    input, gradOutput, gradOutputMean, dotP, gradInput, gradWeight, gradBias, weight, runningMean, runningVar,
    saveMean, saveStd, train, scale, eps);
  THCudaCheck(cudaGetLastError());
}

#undef DeviceTensor3
#undef DeviceTensor1

#endif
