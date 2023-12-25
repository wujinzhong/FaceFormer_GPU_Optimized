# FaceFormer_TensorRT

This is the CUDA/GPU optimized FaceFormer pipeline, many thanks to original developer EvelynFan from here, https://github.com/EvelynFan/FaceFormer, please check tech details there.

## Optimization

Currently we first optimized rendering part performance, and leave AI model conversion to TRT later. We use CUDA/OpenGL interop to speed up the read back framebuffer pixels from OpenGL/pyrender, without leaving GPU, to VPF(VideoProcessingFramework, NVIDIA GPU hardware video encoder/decoder solution for python). To further improve the inference performance, we dev a viewport grid strategy in OpenGL/pyrender which can rendering multi-frames, say 2x3 frames into the same framebuffer, to making rendering pipeline as busy as possible. These 3 optimization schemes makes 3.8X speed up only for rendering/video encoding stages, please check the code in detail. We also split the CUDA/OpenGL interop and viewport grid source code from the baseline, make it easier used in other pipeline.
![image](https://github.com/wujinzhong/FaceFormer_TensorRT/assets/52945455/c9e591d6-8072-4ed7-afb2-017b15ac3f3a)
