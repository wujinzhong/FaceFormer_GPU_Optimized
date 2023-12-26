# FaceFormer_TensorRT

This is the CUDA/GPU optimized FaceFormer pipeline, many thanks to original developer EvelynFan from here, https://github.com/EvelynFan/FaceFormer, please check tech details there.

Please refer to https://github.com/EvelynFan/FaceFormer to download pretrained models from biwi.pth and vocaset.pth. Put the pretrained models under BIWI and VOCASET folders, respectively.

To animate a mesh in BIWI topology, run:

 python demo.py --model_name biwi --wav_path "demo/wav/test.wav" --dataset BIWI --vertice_dim 70110  --feature_dim 128 --period 25 --fps 25 --train_subjects "F2 F3 F4 M3 M4 M5" --test_subjects "F1 F5 F6 F7 F8 M1 M2 M6" --condition M3 --subject M1

## System Config

Refer to original implementation at here for details, https://github.com/EvelynFan/FaceFormer. 

To install CUDA in python, run:

> pip install cuda-python

To install VPF, refer to here, https://github.com/NVIDIA/VideoProcessingFramework.

## Optimization

Currently we first optimized rendering part performance, and leave AI model conversion to TRT later. We use CUDA/OpenGL interop to speed up the read back framebuffer pixels from OpenGL/pyrender, without leaving GPU, to VPF(VideoProcessingFramework, NVIDIA GPU hardware video encoder/decoder solution for python). To further improve the inference performance, we dev a viewport grid strategy in OpenGL/pyrender which can rendering multi-frames, say 2x3 frames into the same framebuffer, to making rendering pipeline as busy as possible. These 3 optimization schemes makes 3.8X speed up only for rendering/video encoding stages, please check the code in detail. We also split the CUDA/OpenGL interop and viewport grid source code from the baseline, make it easier used in other pipeline.
![image](https://github.com/wujinzhong/FaceFormer_TensorRT/assets/52945455/c9e591d6-8072-4ed7-afb2-017b15ac3f3a)
