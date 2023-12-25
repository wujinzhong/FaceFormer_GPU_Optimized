from cuda import cuda, nvrtc, cudart
import numpy as np
import numpy.typing as npt
import nvtx
import PyNvCodec as nvc
from ctypes import c_int, pointer
from OpenGL.GL import *
import os

def cuda_get_error_enum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return (
            name.decode()
            if err == cuda.CUresult.CUDA_SUCCESS
            else "<unknown driver error>"
        )
    elif isinstance(error, cudart.cudaError_t):
        err, name = cudart.cudaGetErrorName(error)
        return (
            name.decode()
            if err == cudart.cudaError_t.cudaSuccess
            else "<unknown runtime error>"
        )
    elif isinstance(error, nvrtc.nvrtcResult):
        err, name = nvrtc.nvrtcGetErrorString(error)
        return (
            name.decode()
            if err == nvrtc.nvrtcResult.NVRTC_SUCCESS
            else "<unknown nvrtc error>"
        )
    else:
        return "<unknown error>"
    
def cuda_check_errors(result):
    if result[0].value:
        raise RuntimeError(
            "CUDA Error code: {} ({})".format(
                result[0].value, cuda_get_error_enum(result[0])
            )
        )

    if len(result) == 2:
        return result[1]
    elif len(result) > 2:
        return result[1:]

def __dtype_to_channel_format(dtype: npt.DTypeLike) -> cudart.cudaChannelFormatKind:
    """helper to convert numpy dtype to cuda channel format kind"""

    if np.issubdtype(dtype, np.floating):
        return cudart.cudaChannelFormatKind.cudaChannelFormatKindFloat
    if np.issubdtype(dtype, np.signedinteger):
        return cudart.cudaChannelFormatKind.cudaChannelFormatKindSigned

    return cudart.cudaChannelFormatKind.cudaChannelFormatKindUnsigned

def create_empty_array( shape: tuple[int, int], dtype: npt.DTypeLike) -> cudart.cudaArray_t:
    """creates and empty 2d array on the device, dtype is used to determine
    memory size of each element in the array"""

    format_kind = __dtype_to_channel_format(dtype)

    rows, cols = shape
    channel_desc = cuda_check_errors(
        cudart.cudaCreateChannelDesc(8 * np.dtype(dtype).itemsize, 0, 0, 0, format_kind)
    )

    return cuda_check_errors(
        cudart.cudaMallocArray(channel_desc, cols, rows, cudart.cudaArrayDefault)
    )

def create_texture(
    array: cudart.cudaArray_t, dtype: npt.DTypeLike
) -> cudart.cudaTextureObject_t:
    """creates a texture object from the given array through which we
    can read the array with some caching benefits, dtype is used to get
    the memory size of a single component"""

    resource_desc = cudart.cudaResourceDesc()
    resource_desc.resType = cudart.cudaResourceType.cudaResourceTypeArray
    resource_desc.res.array.array = array

    read_normalized = dtype == np.uint8 or dtype == np.uint16
    texture_desc = cudart.cudaTextureDesc()
    texture_desc.addressMode[0] = cudart.cudaTextureAddressMode.cudaAddressModeBorder
    texture_desc.addressMode[1] = cudart.cudaTextureAddressMode.cudaAddressModeBorder
    texture_desc.borderColor[0] = 0
    texture_desc.borderColor[1] = 0
    texture_desc.borderColor[2] = 0
    texture_desc.borderColor[3] = 0
    texture_desc.filterMode = cudart.cudaTextureFilterMode.cudaFilterModePoint
    texture_desc.readMode = (
        cudart.cudaTextureReadMode.cudaReadModeNormalizedFloat
        if read_normalized
        else cudart.cudaTextureReadMode.cudaReadModeElementType
    )
    texture_desc.normalizedCoords = 0

    return cuda_check_errors(
        cudart.cudaCreateTextureObject(resource_desc, texture_desc, None)
    )

def create_surface(array: cudart.cudaArray_t) -> cudart.cudaSurfaceObject_t:
    """creates a surface object from the given array through which we can edit the array"""

    resource_desc = cudart.cudaResourceDesc()
    resource_desc.resType = cudart.cudaResourceType.cudaResourceTypeArray
    resource_desc.res.array.array = array

    return cuda_check_errors(cudart.cudaCreateSurfaceObject(resource_desc))

def create_kernel_new(kernel_str: str, name: str, cuDevice: cuda.CUdevice
) -> tuple[cuda.CUfunction, cuda.CUmodule]:
    def ASSERT_DRV(err):
        if isinstance(err, cuda.CUresult):
            if err != cuda.CUresult.CUDA_SUCCESS:
                raise RuntimeError('Cuda Error: {}'.format(err))
        elif isinstance(err, nvrtc.nvrtcResult):
            if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
                raise RuntimeError('Nvrtc Error: {}'.format(err))
        else:
            raise RuntimeError('Unknown error type: {}'.format(err))

    """loades and compiles a cuda kernel from a source file and then retrieves
    the function by name"""

    # Create program
    prog = cuda_check_errors(
        nvrtc.nvrtcCreateProgram(str.encode(kernel_str), bytes(f"{name}.cu", encoding='utf-8'), 0, [], [])
    )
    print(f"prog: {prog}")

    # Get target architecture
    err, major = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice)
    ASSERT_DRV(err)
    err, minor = cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice)
    ASSERT_DRV(err)
    err, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
    ASSERT_DRV(err)
    use_cubin = (nvrtc_minor >= 1)
    prefix = 'sm' if use_cubin else 'compute'
    arch_arg = bytes(f'--gpu-architecture={prefix}_{major}{minor}', 'ascii')

    # Compile program
    opts = [b'--fmad=false', arch_arg]
    err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    ASSERT_DRV(err)

    # Get log from compilation
    err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
    ASSERT_DRV(err)
    log = b' ' * logSize
    err, = nvrtc.nvrtcGetProgramLog(prog, log)
    ASSERT_DRV(err)
    print(log.decode())

    # Get data from compilation
    if use_cubin:
        err, dataSize = nvrtc.nvrtcGetCUBINSize(prog)
        ASSERT_DRV(err)
        data = b' ' * dataSize
        err, = nvrtc.nvrtcGetCUBIN(prog, data)
        ASSERT_DRV(err)
    else:
        err, dataSize = nvrtc.nvrtcGetPTXSize(prog)
        ASSERT_DRV(err)
        data = b' ' * dataSize
        err, = nvrtc.nvrtcGetPTX(prog, data)
        ASSERT_DRV(err)

    # Load data as module data and retrieve function
    data = np.char.array(data)
    err, module = cuda.cuModuleLoadData(data)
    ASSERT_DRV(err)
    err, kernel = cuda.cuModuleGetFunction(module, name.encode())
    ASSERT_DRV(err)

    module = cuda_check_errors(cuda.cuModuleLoadData(np.char.array(data)))
    cuda_check_errors(nvrtc.nvrtcDestroyProgram(prog))

    return cuda_check_errors(cuda.cuModuleGetFunction(module, name.encode())), module

def create_kernel(
    source_file: str, name: str, device: cuda.CUdevice
) -> tuple[cuda.CUfunction, cuda.CUmodule]:
    """loades and compiles a cuda kernel from a source file and then retrieves
    the function by name"""

    with open(source_file, "r") as file:
        source = file.read()

    program = cuda_check_errors(
        nvrtc.nvrtcCreateProgram(source.encode(), b"temp.cu", 0, [], [])
    )

    cuda_home = os.getenv("CUDA_HOME")
    if cuda_home is None:
        raise RuntimeError("CUDA_HOME not set")

    include_dir = os.path.join(cuda_home, "include")
    include_arg = f"--include-path={include_dir}".encode()

    major = cuda_check_errors(
        cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, device
        )
    )
    minor = cuda_check_errors(
        cudart.cudaDeviceGetAttribute(
            cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, device
        )
    )
    major = 8
    minor = 0
    arch_arg = f"--gpu-architecture=compute_{major}{minor}".encode()
    arch_arg = f"".encode()
    print(f"arch_arg: {arch_arg}")

    opts = [b"--fmad=true", include_arg, b"--std=c++17", b"-default-device"]

    try:
        cuda_check_errors(nvrtc.nvrtcCompileProgram(program, len(opts), opts))
    except RuntimeError:
        log_size = cuda_check_errors(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * log_size  # type: ignore

        cuda_check_errors(nvrtc.nvrtcGetProgramLog(program, log))
        raise RuntimeError(log.decode())

    data_size = cuda_check_errors(nvrtc.nvrtcGetPTXSize(program))
    data = b" " * data_size  # type: ignore

    cuda_check_errors(nvrtc.nvrtcGetPTX(program, data))
    print(f"data: {data}")
    module = cuda_check_errors(cuda.cuModuleLoadData(np.char.array(data)))
    cuda_check_errors(nvrtc.nvrtcDestroyProgram(program))

    return cuda_check_errors(cuda.cuModuleGetFunction(module, name.encode())), module

def create_cuda_context() -> tuple[cuda.CUdevice, cuda.CUcontext]:
    """creates a cuda context on the first available device"""

    cuda_check_errors(cuda.cuInit(0))
    gpu = 0
    device = cuda_check_errors(cuda.cuDeviceGet(gpu))
    context = cuda_check_errors(cuda.cuCtxCreate(0, device))

    return device, context

class VPF_Color_Converter:
    """
    Colorspace conversion chain.
    """
    def __init__(self, width: int, height: int, context: int, stream: int):
        #self.gpu_id = gpu_id
        self.w = width
        self.h = height
        self.chain = []
        self.context = context
        self.stream = stream

    def add(self, src_fmt: nvc.PixelFormat, dst_fmt: nvc.PixelFormat) -> None:
        self.chain.append(
            nvc.PySurfaceConverter(self.w, self.h, src_fmt, dst_fmt, self.context, self.stream)
        )

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        return surf.Clone() #surf.Clone(self.gpu_id)
    
def video_nvc_surfaces_to_VPF(nvc_surfaces, path, fps=20, torch_stream=None):
    rng = nvtx.start_range(message="VPF0", color="red")

    width = nvc_surfaces[0].Width()
    height = nvc_surfaces[0].Height()
    print(f"width, height: {width, height}")

    gt_res_change = 47
    gt_is_vfr = False
    gt_pix_fmt = nvc.PixelFormat.NV12
    gt_framerate = 30
    gt_num_frames = 96
    gt_timebase = 8.1380e-5
    gt_color_space = nvc.ColorSpace.BT_709
    gt_color_range = nvc.ColorRange.MPEG
    gpu_id = 0

    rng_nv12 = nvtx.start_range(message="nv12", color='blue')
    
    # Retain primary CUDA device context and create separate stream per thread.
    _, c_ctx = cuda.cuCtxGetCurrent()
    print(f"c_ctx: {c_ctx}")
    _, c_str = cuda.cuStreamCreate(0) #torch.cuda.current_stream().cuda_stream
    
    to_nv12 = VPF_Color_Converter(width, height, c_ctx, c_str)
    to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
    to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
    to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
    nvtx.end_range(rng_nv12)
    
    res = str(width) + "x" + str(height)
    encFrame = np.ndarray(shape=(0), dtype=np.uint8)
    
    dstFile = open(path, "wb")

    pixel_format = nvc.PixelFormat.NV12
    profile = "high"
    surfaceformat = "nv12"
    if surfaceformat == 'yuv444':
        pixel_format = nvc.PixelFormat.YUV444
        profile = "high_444"
    elif surfaceformat == 'yuv444_10bit':
        pixel_format = nvc.PixelFormat.YUV444_10bit
        profile = "high_444_10bit"
    elif surfaceformat == 'yuv420_10bit':
        pixel_format = nvc.PixelFormat.YUV420_10bit
        profile = "high_420_10bit"
    
    rng_nvEnc = nvtx.start_range(message="nvc", color='blue')
    nvEnc = nvc.PyNvEncoder(
        {
            "preset": "P4",
            "tuning_info": "high_quality",
            "codec": "h264", #"hevc"
            "profile": profile, #"high"
            "s": res,
            "bitrate": "5M",
            'fps': f'{fps}', #20
        },
        #gpu_id,
        c_ctx,
        c_str,
        pixel_format, #nvc.PixelFormat.NV12
    )
    print(f"nvEnc: {nvEnc}")
    nvtx.end_range(rng_nvEnc)

    frames_sent = 0
    frames_recv = 0

    nvtx.end_range(rng)
    rng = nvtx.start_range(message="VPF1", color="red")
    while frames_sent<len(nvc_surfaces):            
        dec_surf = nvc_surfaces[frames_sent]
        if not dec_surf or dec_surf.Empty():
            print("break")
            break
        dec_surf = to_nv12.run(dec_surf)
        
        frames_sent += 1

        nvEnc.EncodeSingleSurface(dec_surf, encFrame)
        if encFrame.size:
            frames_recv += 1

            byteArray = bytearray(encFrame)
            print(f"len(byteArray): {len(byteArray)}")
            dstFile.write(byteArray)

    while True:
        success = nvEnc.FlushSinglePacket(encFrame)
        if success and encFrame.size:
            frames_recv += 1
        else:
            break

    assert frames_sent==frames_recv
    print(f"encode frames {frames_sent}")
    nvtx.end_range(rng)

    return None

def video_torch_tensor_to_VPF(tensor, path, duration = 120, loop = 0, optimize = True, torch_stream=None,
                        testIdx=0):
    rng = nvtx.start_range(message="VPF0", color="red")
    tensor = tensor.permute(1,2,3,0)
    images = tensor.unbind(dim = 0)
    images_gpu = [(image*255).to(torch.uint8) for image in images]
    
    #for i, image in enumerate(images_gpu):
    #    print(f"image{i}: {image.dtype, image.shape}")

    if True: #VPF imp
        # Ground truth information about input video
        #gt_file = join(dirname(__file__), "test.mp4")
        #gt_file_res_change = join(dirname(__file__), "test_res_change.h264")
        gt_width = images_gpu[0].shape[1]
        gt_height = images_gpu[0].shape[0]
        gt_res_change = 47
        gt_is_vfr = False
        gt_pix_fmt = nvc.PixelFormat.NV12
        gt_framerate = 30
        gt_num_frames = 96
        gt_timebase = 8.1380e-5
        gt_color_space = nvc.ColorSpace.BT_709
        gt_color_range = nvc.ColorRange.MPEG
        gpu_id = 0

        #rawFrame = np.random.randint(0, 255, size=(gt_height, gt_width, 3), dtype=np.uint8)
        #nvDec = nvc.PyNvDecoder(gt_file, gpu_id)
        #c_nvUpl = nvc.PyFrameUploader(
        #        gt_width, gt_height, nvc.PixelFormat.RGB_PLANAR, c_ctx, c_str
        #    )

        rng_nv12 = nvtx.start_range(message="nv12", color='blue')
        
        # Retain primary CUDA device context and create separate stream per thread.
        _, c_ctx = cuda.cuCtxGetCurrent()
        c_str = torch_stream #torch.cuda.current_stream().cuda_stream
        
        to_nv12 = VPF_Color_Converter(gt_width, gt_height, c_ctx, c_str.cuda_stream)
        to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
        nvtx.end_range(rng_nv12)

        
        res = str(gt_width) + "x" + str(gt_height)
        encFrame = np.ndarray(shape=(0), dtype=np.uint8)
        
        dstFile = open(path.split(".gif")[0]+f"_vpf_{testIdx}.mp4", "wb")

        pixel_format = nvc.PixelFormat.NV12
        profile = "high"
        surfaceformat = "nv12"
        if surfaceformat == 'yuv444':
            pixel_format = nvc.PixelFormat.YUV444
            profile = "high_444"
        elif surfaceformat == 'yuv444_10bit':
            pixel_format = nvc.PixelFormat.YUV444_10bit
            profile = "high_444_10bit"
        elif surfaceformat == 'yuv420_10bit':
            pixel_format = nvc.PixelFormat.YUV420_10bit
            profile = "high_420_10bit"
        
        rng_nvEnc = nvtx.start_range(message="nvc", color='blue')
        nvEnc = nvc.PyNvEncoder(
            {
                "preset": "P4",
                "tuning_info": "high_quality",
                "codec": "h264", #"hevc"
                "profile": profile, #"high"
                "s": res,
                "bitrate": "5M",
                'fps': '1', 
            },
            #gpu_id,
            c_ctx,
            c_str.cuda_stream,
            pixel_format, #nvc.PixelFormat.NV12
        )
        print(f"nvEnc: {nvEnc}")
        nvtx.end_range(rng_nvEnc)

        frames_sent = 0
        frames_recv = 0

        nvtx.end_range(rng)
        rng = nvtx.start_range(message="VPF1", color="red")
        while frames_sent<len(images_gpu):
            rawSurface = cuda_tensor_to_surface( images_gpu[frames_sent], gpu_id=0, torch_stream=torch_stream )
            
            dec_surf = rawSurface
            if not dec_surf or dec_surf.Empty():
                break
            dec_surf = to_nv12.run(dec_surf)
            
            frames_sent += 1

            nvEnc.EncodeSingleSurface(dec_surf, encFrame)
            if encFrame.size:
                frames_recv += 1

                byteArray = bytearray(encFrame)
                dstFile.write(byteArray)

        while True:
            success = nvEnc.FlushSinglePacket(encFrame)
            if success and encFrame.size:
                frames_recv += 1
            else:
                break

        assert frames_sent==frames_recv
        print(f"encode frames {frames_sent}")
        nvtx.end_range(rng)

    return None

def cuda_array_to_nvc_surface_backup( input_texture, gpu_id, height, width, channel ):
        surf_dst = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, width, height, gpu_id)
        assert not surf_dst.Empty()
        print(f"surf_dst: {surf_dst}")
        dst_plane = surf_dst.PlanePtr()

        ##cuda.cuMemcpyDtoDAsync(dst_plane.GpuMem(), tensor.data_ptr(), tensor_h * tensor_w * tensor_c, torch_stream.cuda_stream)
        #cuda_check_errors(
        #    cuda.cuLaunchKernel(
        #        self.kernel_cuda_array_to_nvc_surface,
        #        int(np.ceil(width / 32)),
        #        int(np.ceil(height / 32)),
        #        1,
        #        32,
        #        32,
        #        1,
        #        0,
        #        0,
        #        (
        #            (dst_plane.GpuMem(), input_texture, width, height),
        #            (None, None, c_int, c_int),
        #        ),
        #        0,
        #    )
        #)
        
        return surf_dst

        import pycuda.driver as pycuda

        pycuda.memcpy_dtod_async(dst_plane.GpuMem(), tensor.data_ptr(), 
                                        tensor_h * tensor_w * tensor_c)

        memcpy_2d = pycuda.Memcpy2D()
        memcpy_2d.width_in_bytes = tensor_w * tensor_c
        memcpy_2d.src_pitch = tensor_w * tensor_c
        memcpy_2d.dst_pitch = dst_plane.Pitch()
        memcpy_2d.width = tensor_h
        memcpy_2d.height = tensor_h
        print(f"memcpy pos0")
        memcpy_2d.set_src_device(tensor.data_ptr())
        print(f"memcpy pos1")
        memcpy_2d.set_dst_device(dst_plane.GpuMem())
        print(f"memcpy pos2")
        print(f"memcpy pos2, torch_stream.cuda_stream: {torch_stream.cuda_stream}")
        memcpy_2d(torch_stream.cuda_stream)
        print(f"memcpy pos3")

def cuda_tensor_to_surface( tensor, gpu_id, torch_stream=None ):
    tensor_h = tensor.shape[0]
    tensor_w = tensor.shape[1]
    tensor_c = tensor.shape[2] #channel
    surf_dst = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, tensor_w, tensor_h, gpu_id)

    #print(f"tensor: {tensor.shape, tensor.device, tensor.dtype, tensor.stride(), tensor.storage_offset(), tensor.data_ptr()}")

    assert not surf_dst.Empty()
    dst_plane = surf_dst.PlanePtr()

    cuda.cuMemcpyDtoDAsync(dst_plane.GpuMem(), tensor.data_ptr(), tensor_h * tensor_w * tensor_c, torch_stream.cuda_stream)
    trt_util_2.synchronize(torch_stream)
    return surf_dst

    import pycuda.driver as pycuda

    pycuda.memcpy_dtod_async(dst_plane.GpuMem(), tensor.data_ptr(), 
                                    tensor_h * tensor_w * tensor_c)

    memcpy_2d = pycuda.Memcpy2D()
    memcpy_2d.width_in_bytes = tensor_w * tensor_c
    memcpy_2d.src_pitch = tensor_w * tensor_c
    memcpy_2d.dst_pitch = dst_plane.Pitch()
    memcpy_2d.width = tensor_h
    memcpy_2d.height = tensor_h
    print(f"memcpy pos0")
    memcpy_2d.set_src_device(tensor.data_ptr())
    print(f"memcpy pos1")
    memcpy_2d.set_dst_device(dst_plane.GpuMem())
    print(f"memcpy pos2")
    print(f"memcpy pos2, torch_stream.cuda_stream: {torch_stream.cuda_stream}")
    memcpy_2d(torch_stream.cuda_stream)
    print(f"memcpy pos3")

def select_EGL_device_to():
    from pyrender.platforms import egl
    device_id = int(os.environ.get('EGL_DEVICE_ID', '1'))
    egl_device = egl.get_device_by_index(device_id)
    print(f"device_id: {device_id}, name: {egl_device.name}")
    
    viewport_width = 800
    viewport_height = 800
    _platform = egl.EGLPlatform(viewport_width,
                                viewport_height,
                                device=egl_device)
    
def print_opengl_vendor_info():
    print(f"glGetString(GL_VENDOR): {glGetString(GL_VENDOR)}")
    print(f"glGetString(GL_RENDERER): {glGetString(GL_RENDERER)}")

def VPF_encoding_case_1():
    #######################################################################
    ###################     must init EGL/OpenGL first  ###################
    #######################################################################
    viewport_height = 800
    viewport_width = 800
    # Generate standard buffer
    _main_cb, _main_db = glGenRenderbuffers(2)

    glBindRenderbuffer(GL_RENDERBUFFER, _main_cb)
    glRenderbufferStorage(
        GL_RENDERBUFFER, GL_RGBA,
        viewport_width, viewport_height
    )

    glBindRenderbuffer(GL_RENDERBUFFER, _main_db)
    glRenderbufferStorage(
        GL_RENDERBUFFER, GL_DEPTH_COMPONENT24,
        viewport_width, viewport_height
    )

    _main_fb = glGenFramebuffers(1)
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, _main_fb)
    glFramebufferRenderbuffer(
        GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
        GL_RENDERBUFFER, _main_cb
    )
    glFramebufferRenderbuffer(
        GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
        GL_RENDERBUFFER, _main_db
    )

    #######################################################################
    ##########         must init CUDA context after OpenGL init       #####
    #######################################################################
    cuda_device, cuda_context = create_cuda_context()

    cuda_check_errors(cuda.cuCtxSetCurrent(cuda_context))
    
    shape = (viewport_height, viewport_width)
    # allocate an array for the output of the kernel
    output_array = create_empty_array(shape, np.uint32)
    # and bind a surface to it
    output_surface = create_surface(output_array)

    print(f"cuda_device, cuda_context: {cuda_device, cuda_context}")
    
    kernel_cuda_array_to_surface, module_cuda_array_to_surface = create_kernel_new(cuda_array_to_surface_str, "cuda_array_to_surface", cuda_device)
    print(f"kernel_cuda_array_to_surface: {kernel_cuda_array_to_surface}")
    print(f"module_cuda_array_to_surface: {module_cuda_array_to_surface}")

    kernel_cuda_array_to_nvc_surface, module_cuda_array_to_nvc_surface = create_kernel_new(cuda_array_to_nvc_surface_str, "cuda_array_to_nvc_surface", cuda_device)
    print(f"kernel_cuda_array_to_nvc_surface: {kernel_cuda_array_to_nvc_surface}")
    print(f"module_cuda_array_to_nvc_surface: {module_cuda_array_to_nvc_surface}")

    kernel_nvc_surface_to_surface, module_nvc_surface_to_surface = create_kernel_new(nvc_surface_to_surface_str, "nvc_surface_to_surface", cuda_device)
    print(f"kernel_nvc_surface_to_surface: {kernel_nvc_surface_to_surface}")
    print(f"module_nvc_surface_to_surface: {module_nvc_surface_to_surface}")

    image = cuda_check_errors(
    cudart.cudaGraphicsGLRegisterImage(
        _main_cb,
        GL_RENDERBUFFER,
        cudart.cudaGraphicsRegisterFlags.cudaGraphicsRegisterFlagsWriteDiscard,
    ))
    
    # register the renderbuffer as image in cuda
    #print(f"_main_cb: {_main_cb}")
    
    cuda_check_errors(cuda.cuCtxSetCurrent(cuda_context))

    # map the resource, could be seen as cuda taking ownership of it
    # opengl operations on it after this call are undefined
    cuda_check_errors(cudart.cudaGraphicsMapResources(1, image, None))
    #print(f"image: {image}")
    # get the underlying array
    input_array = cuda_check_errors(
        cudart.cudaGraphicsSubResourceGetMappedArray(image, 0, 0)
    )

    # and bind a texture to the mapped array
    input_texture = create_texture(input_array, np.uint8)

    #print(f"input_texture: {input_texture}")
    #print(f"output_surface: {output_surface}")

    viewport_grid_x = 2
    viewport_grid_y = 3

    width = viewport_width * viewport_grid_x
    height = viewport_height * viewport_grid_y

    gpu_id = 0
    surf_dsts = []
    for gridY in range(viewport_grid_y):
        for gridX in range(viewport_grid_x):
            #print(f"gridY, gridX: {gridY, gridX}")
            #print(f"viewport_grid_x, viewport_grid_y: {viewport_grid_x, viewport_grid_y}")
            surf_dst = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, width//viewport_grid_x, height//viewport_grid_y, gpu_id)
            assert not surf_dst.Empty()
            dst_plane0 = surf_dst.PlanePtr(0)

            #print(f"surf_dst: {surf_dst}")
            #dst_plane1 = surf_dst.PlanePtr(1)
            #dst_plane2 = surf_dst.PlanePtr(2)
            #print(f"cuda.CUdeviceptr(dst_plane0.GpuMem()): {cuda.CUdeviceptr(dst_plane0.GpuMem())}")
            #print(f"dst_plane0: {dst_plane0.GpuMem(), dst_plane0.Width(), dst_plane0.Height(), dst_plane0.Pitch(), dst_plane0}")
            #print(f"dst_plane1: {dst_plane1.GpuMem(), dst_plane0.Width(), dst_plane0.Height(), dst_plane0.Pitch(), dst_plane1}")
            #print(f"dst_plane2: {dst_plane2.GpuMem(), dst_plane0.Width(), dst_plane0.Height(), dst_plane0.Pitch(), dst_plane2}")

            assert (height//viewport_grid_y*3)==dst_plane0.Height()
            cuda_check_errors(
                cuda.cuLaunchKernel(
                    kernel_cuda_array_to_nvc_surface,
                    int(np.ceil(width / viewport_grid_x / 32)),
                    int(np.ceil(height / viewport_grid_y / 32)),
                    1,
                    32,
                    32,
                    1,
                    0,
                    0,
                    (
                        (input_texture, cuda.CUdeviceptr(dst_plane0.GpuMem()), 
                            width, height, 
                            surf_dst.Width(), surf_dst.Height(), dst_plane0.Pitch(),
                            gridX, gridY),
                        (None, None, c_int, c_int, c_int, c_int, c_int, c_int, c_int),
                    ),
                    0,
                )
            )
            if False:
                ## launch the edge detection kernel
                #cuda_check_errors(
                #    cuda.cuLaunchKernel(
                #        kernel_cuda_array_to_surface,
                #        int(np.ceil(width / 32)),
                #        int(np.ceil(height / 32)),
                #        1,
                #        32,
                #        32,
                #        1,
                #        0,
                #        0,
                #        (
                #            (input_texture, output_surface, width, height),
                #            (None, None, c_int, c_int),
                #        ),
                #        0,
                #    )
                #)

                # launch the edge detection kernel
                cuda_check_errors(
                    cuda.cuLaunchKernel(
                        kernel_nvc_surface_to_surface,
                        int(np.ceil(width / 32)),
                        int(np.ceil(height / 32)),
                        1,
                        32,
                        32,
                        1,
                        0,
                        0,
                        (
                            (cuda.CUdeviceptr(dst_plane0.GpuMem()), output_surface, width, height, dst_plane0.Width(), dst_plane0.Height(), dst_plane0.Pitch()),
                            (None, None, c_int, c_int, c_int, c_int, c_int),
                        ),
                        0,
                    )
                )

                # prep a host array to copy the result into
                #print(f"input_array: {input_array.shape}")
                img_numpy = np.empty((*shape, 4), dtype=np.uint8)
                # copy the result from the ouput array to the host so we can look at it
                cuda_check_errors(
                    cudart.cudaMemcpy2DFromArray(
                        img_numpy.ctypes.data,
                        img_numpy.shape[1] * img_numpy.itemsize * 4,
                        output_array,
                        0,
                        0,
                        img_numpy.shape[1] * img_numpy.itemsize * 4,
                        img_numpy.shape[0],
                        cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    )
                )
                #img = Image.fromarray(img_numpy, "RGBA")
                #img.save(f"./demo/png/img_numpy{g_img_idx}.png")
                #g_img_idx += 1
            
            surf_dsts.append(surf_dst)
    # unmap the resource, operations on it in cuda after this call are undefined
    # but opengl can use it again safely
    cuda_check_errors(cudart.cudaGraphicsUnmapResources(1, image, None))

    ##print(f"color_im: {color_im.shape, color_im.dtype}")
    #img_numpy_2 = img_numpy[:, :, 0:3]
    #color_im = img_numpy_2

edge = '''\
extern "C" __global__
void edge(
    cudaTextureObject_t render,
    cudaSurfaceObject_t output,
    int width,
    int height
) {
    unsigned int sx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sy = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int stridex = blockDim.x * gridDim.x;
    unsigned int stridey = blockDim.y * gridDim.y;

    for (int x = sx; x < width; x += stridex) {
        for (int y = sy; y < height; y += stridey) {
            float4 top_left = tex2D<float4>(render, x - 1, y - 1);
            float4 top_center = tex2D<float4>(render, x, y - 1);
            float4 top_right = tex2D<float4>(render, x + 1, y - 1);

            float4 center_left = tex2D<float4>(render, x - 1, y);
            // float4 center_center = tex2D<float4>(render, x, y);
            float4 center_right = tex2D<float4>(render, x + 1, y);

            float4 bottom_left = tex2D<float4>(render, x - 1, y + 1);
            float4 bottom_center = tex2D<float4>(render, x, y + 1);
            float4 bottom_right = tex2D<float4>(render, x + 1, y + 1);

            // [-1  0  1]  [-1 -1 -1]
            // [-1  0  1]  [ 0  0  0]
            // [-1  0  1]  [ 1  1  1]
            float mag_vx = - top_left.x - center_left.x - bottom_left.x
                + top_right.x + center_right.x + bottom_right.x;

            float mag_vy = - top_left.y - center_left.y - bottom_left.y
                + top_right.y + center_right.y + bottom_right.y;

            float mag_vz = - top_left.z - center_left.z - bottom_left.z
                + top_right.z + center_right.z + bottom_right.z;

            float mag_hx = - top_left.x - top_center.x - top_right.x
                + bottom_left.x + bottom_center.x + bottom_right.x;

            float mag_hy = - top_left.y - top_center.y - top_right.y
                + bottom_left.y + bottom_center.y + bottom_right.y;

            float mag_hz = - top_left.z - top_center.z - top_right.z
                + bottom_left.z + bottom_center.z + bottom_right.z;

            float mag = sqrt(mag_hx * mag_hx + mag_vx * mag_vx
                            + mag_hy * mag_hy + mag_vy * mag_vy
                            + mag_hz * mag_hz + mag_vz * mag_vz) / 3;

            // convert the float value back to rgba uint8 (grayscale)
            unsigned char mag_char = (unsigned char) (mag * 255);
            uchar4 mag_char4 = make_uchar4(mag_char, mag_char, mag_char, 255);
            surf2Dwrite(mag_char4, output, 4 * x, y, cudaBoundaryModeZero);
        }
    }
}
'''

cuda_array_to_surface_str = '''\
extern "C" __global__
void cuda_array_to_surface(
    cudaTextureObject_t render,
    cudaSurfaceObject_t output,
    int width,
    int height
) {
    unsigned int sx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sy = blockIdx.y * blockDim.y + threadIdx.y;
    int x = sx;
    int y = sy;
    if ( x<width && y<height )
    {
        float4 center_center = tex2D<float4>(render, x, y);
        // convert the float value back to rgba uint8 (grayscale)
        unsigned char r = (unsigned char) (center_center.x * 255);
        unsigned char g = (unsigned char) (center_center.y * 255);
        unsigned char b = (unsigned char) (center_center.z * 255);
        uchar4 mag_char4 = make_uchar4(r, g, b, 255);
        surf2Dwrite(mag_char4, output, 4 * x, y, cudaBoundaryModeZero);
    }            
}
'''

cuda_array_to_nvc_surface_str = '''\
extern "C" __global__
void cuda_array_to_nvc_surface(
    cudaTextureObject_t render,
    unsigned char* output,
    int width,
    int height,
    int surfaceWidth, 
    int surfaceHeight,
    int surfacePitch,
    int gridX, 
    int gridY
) {
    unsigned int sx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sy = blockIdx.y * blockDim.y + threadIdx.y;
    int x = sx + gridX*surfaceWidth;
    int y = sy + gridY*surfaceHeight;
    if ( x<width && y<height )
    {
        float4 center_center = tex2D<float4>(render, x, y);
        // convert the float value back to rgba uint8 (grayscale)
        unsigned char r = (unsigned char) (center_center.x * 255);
        unsigned char g = (unsigned char) (center_center.y * 255);
        unsigned char b = (unsigned char) (center_center.z * 255);

        int flipY = surfaceHeight-1-sy;
        output[0*(surfacePitch*surfaceHeight)+flipY*surfacePitch+sx] = r;
        output[1*(surfacePitch*surfaceHeight)+flipY*surfacePitch+sx] = g;
        output[2*(surfacePitch*surfaceHeight)+flipY*surfacePitch+sx] = b;
    }            
}
'''

nvc_surface_to_surface_str = '''\
extern "C" __global__
void nvc_surface_to_surface(
    unsigned char* input,
    cudaSurfaceObject_t output,
    int width,
    int height,
    int surfaceWidth, 
    int surfaceHeight,
    int surfacePitch
) {
    unsigned int sx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sy = blockIdx.y * blockDim.y + threadIdx.y;
    int x = sx;
    int y = sy;
    if ( x<width && y<height )
    {
        int flipY = height-1-y;
        unsigned char r = input[0*(surfacePitch*height)+flipY*surfacePitch+x];
        unsigned char g = input[1*(surfacePitch*height)+flipY*surfacePitch+x];
        unsigned char b = input[2*(surfacePitch*height)+flipY*surfacePitch+x];

        uchar4 mag_char4 = make_uchar4(r, g, b, 255);
        surf2Dwrite(mag_char4, output, 4 * x, y, cudaBoundaryModeZero);
    }            
}
'''