import os,sys,shutil,argparse,copy,pickle
os.environ['PYOPENGL_PLATFORM'] = 'egl' # egl, osmesa
from cuda import cuda, cudart

import numpy as np
import scipy.io.wavfile as wav
import librosa
import math,scipy
from faceformer import Faceformer
from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import tempfile
from subprocess import call
import pyrender
from psbody.mesh import Mesh
import trimesh
import nvtx

from InferenceUtil import (
    Memory_Manager,
    TorchUtil,
    NVTXUtil,
    SynchronizeUtil,
    check_onnx,
    build_TensorRT_engine_CLI,
    TRT_Engine,
    USE_TRT,
    USE_WARM_UP
)
import CodecUtil

@torch.no_grad()
def test_model(args, mm, torchutil):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    with NVTXUtil("load model", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        #build model
        model = Faceformer(args)
        model.load_state_dict(torch.load(os.path.join(args.dataset, '{}.pth'.format(args.model_name))))
        model = model.to(torch.device(args.device))
        model.eval()

    with NVTXUtil("load template", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        template_file = os.path.join(args.dataset, args.template_path)
        with open(template_file, 'rb') as fin:
            templates = pickle.load(fin,encoding='latin1')

    with NVTXUtil("preprocess", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        train_subjects_list = [i for i in args.train_subjects.split(" ")]

        one_hot_labels = np.eye(len(train_subjects_list))
        iter = train_subjects_list.index(args.condition)
        one_hot = one_hot_labels[iter]
        one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
        one_hot = torch.FloatTensor(one_hot).to(device=args.device)

        temp = templates[args.subject]
                
        template = temp.reshape((-1))
        template = np.reshape(template,(-1,template.shape[0]))
        template = torch.FloatTensor(template).to(device=args.device)

    with NVTXUtil("rosa load", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        wav_path = args.wav_path
        test_name = os.path.basename(wav_path).split(".")[0]
        speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    
    with NVTXUtil("Wav2Vec load", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    with NVTXUtil("Wav2Vec run", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    with NVTXUtil("predict", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        prediction = model.predict(audio_feature, template, one_hot)
        prediction = prediction.squeeze() # (seq_len, V*3)
        np.save(os.path.join(args.result_path, test_name), prediction.detach().cpu().numpy())

# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,render_mesh, t_center, rot=np.zeros(3), tex_img=None, z_offset=0, offscreen_renderer=None, frustum=None,
viewport_offset_x=None, viewport_offset_y=None, mm=None,
viewport_grid_x=1, viewport_grid_y=1):
    with NVTXUtil("render_mesh_helper_0", "red", mm):
        if args.dataset == "BIWI":
            camera_params = {'c': np.array([400, 400]),
                            'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                            'f': np.array([4754.97941935 / 8, 4754.97941935 / 8])}
        elif args.dataset == "vocaset":
            camera_params = {'c': np.array([400, 400]),
                            'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
                            'f': np.array([4754.97941935 / 2, 4754.97941935 / 2])}

        #with NVTXUtil("material", "red", mm):
        #    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        #                alphaMode='BLEND',
        #                baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        #                metallicFactor=0.8, 
        #                roughnessFactor=0.8 
        #            )

        #with NVTXUtil("Mesh", "red", mm):
        #    mesh_copy = Mesh(mesh.v, mesh.f)
        #    mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
        #    intensity = 2.0
        #    rgb_per_v = None
        intensity = 2.0
        #with NVTXUtil("trimesh", "red", mm):
        #    print( f"tri_mesh = trimesh.Trimesh" )
        #    tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v)
        #with NVTXUtil("render_mesh", "red", mm):
        #    print( f"render_mesh = pyrender.Mesh.from_trimesh" )
        #    render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)
        
        render_mesh = render_mesh


    with NVTXUtil("scene create", "red", mm):
        if args.background_black:
            scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
        else:
            scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])
        camera = pyrender.IntrinsicsCamera(fx=camera_params['f'][0],
                                        fy=camera_params['f'][1],
                                        cx=camera_params['c'][0],
                                        cy=camera_params['c'][1],
                                        znear=frustum['near'],
                                        zfar=frustum['far'])

    with NVTXUtil("scene add", "red", mm):
        rng = nvtx.start_range(message="scene.add", color="blue")
        scene.add(render_mesh, pose=np.eye(4))
        nvtx.end_range(rng)

        camera_pose = np.eye(4)
        camera_pose[:3,3] = np.array([0, 0, 1.0-z_offset])
        scene.add(camera, pose=[[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 1],
                                [0, 0, 0, 1]])

        angle = np.pi / 6.0
        pos = camera_pose[:3,3]
        light_color = np.array([1., 1., 1.])
        light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

        light_pose = np.eye(4)
        light_pose[:3,3] = pos
        scene.add(light, pose=light_pose.copy())
        
        light_pose[:3,3] = cv2.Rodrigues(np.array([angle, 0, 0]))[0].dot(pos)
        scene.add(light, pose=light_pose.copy())

        light_pose[:3,3] =  cv2.Rodrigues(np.array([-angle, 0, 0]))[0].dot(pos)
        scene.add(light, pose=light_pose.copy())

        light_pose[:3,3] = cv2.Rodrigues(np.array([0, -angle, 0]))[0].dot(pos)
        scene.add(light, pose=light_pose.copy())

        light_pose[:3,3] = cv2.Rodrigues(np.array([0, angle, 0]))[0].dot(pos)
        scene.add(light, pose=light_pose.copy())

    with NVTXUtil("scene rendering", "red", mm):
        flags = pyrender.RenderFlags.SKIP_CULL_FACES
        try:
            offscreen_renderer._renderer.set_viewport_offset(   
                                            viewport_offset_x*frustum["width"], 
                                            viewport_offset_y*frustum["height"],
                                            frustum["width"],
                                            frustum["height"],
                                            viewport_grid_x=viewport_grid_x, 
                                            viewport_grid_y=viewport_grid_y )
            offscreen_renderer.render_just_push(scene)
        except:
            print('pyrender: Failed rendering frame')
            color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return scene

def render_mesh_helper_get_rendered_output_Grid(offscreen_renderer=None, scene=None):
    color, _ = offscreen_renderer.render_read_back(scene)

    return color[..., ::-1], None

def render_mesh_helper_get_rendered_output_CUDA_OpenGL_Interop(offscreen_renderer=None, scene=None):
    color, _, surf_dst = offscreen_renderer.render_read_back_CUDA_OpenGL_Interop(scene)

    return color[..., ::-1], surf_dst

import numpy as np
import time
from tqdm import trange
from math import sqrt
from numba import cuda as numba_cuda

@numba_cuda.jit
def calculate_weighted_normals(normals, vertices, facets, len):
    tid = numba_cuda.grid(1)
    if tid<len:
        f = facets[tid]
        v0 = vertices[f[0]]
        v1 = vertices[f[1]]
        v2 = vertices[f[2]]

        v1_0 = v1[0]-v0[0]
        v1_1 = v1[1]-v0[1]
        v1_2 = v1[2]-v0[2]

        v2_0 = v2[0]-v0[0]
        v2_1 = v2[1]-v0[1]
        v2_2 = v2[2]-v0[2]

        v0_0 = v1_1*v2_2 - v1_2*v2_1
        v0_1 = v1_2*v2_0 - v1_0*v2_2
        v0_2 = v1_0*v2_1 - v1_1*v2_0

        numba_cuda.atomic.add(normals, (f[0],0), v0_0)
        numba_cuda.atomic.add(normals, (f[0],1), v0_1)
        numba_cuda.atomic.add(normals, (f[0],2), v0_2)

        numba_cuda.atomic.add(normals, (f[1],0), v0_0)
        numba_cuda.atomic.add(normals, (f[1],1), v0_1)
        numba_cuda.atomic.add(normals, (f[1],2), v0_2)

        numba_cuda.atomic.add(normals, (f[2],0), v0_0)
        numba_cuda.atomic.add(normals, (f[2],1), v0_1)
        numba_cuda.atomic.add(normals, (f[2],2), v0_2) 

@numba_cuda.jit
def normalize_normals(normals, len):
    tid = numba_cuda.grid(1)
    if tid<len:
        n = normals[tid]
        d = n[0]*n[0] + n[1]*n[1] + n[2]*n[2]
        d = sqrt(d)
        n_0 = n[0]/d
        n_1 = n[1]/d
        n_2 = n[2]/d
        normals[tid][0] = n_0
        normals[tid][1] = n_1
        normals[tid][2] = n_2

@numba_cuda.jit
def zeros_vector3f(normals, len):
    tid = numba_cuda.grid(1)
    if tid<len:
        normals[tid][0] = 0.0
        normals[tid][1] = 0.0
        normals[tid][2] = 0.0

def set_to_zeros( normals_cuda ):
    thread_num = 256
    numba_cuda.synchronize()
    zeros_vector3f[(normals_cuda.shape[0]+thread_num-1)//thread_num,thread_num](normals_cuda, normals_cuda.shape[0])
    numba_cuda.synchronize()

def calculate_normals( normals_cuda, vertices, facets, facets_cuda=None ):
    #normals_cuda = numba_cuda.to_device(normals)
    vertices_cuda = numba_cuda.to_device(vertices)
    if facets_cuda is None:
        facets_cuda = numba_cuda.to_device(facets)

    set_to_zeros(normals_cuda)
    
    #print(f"vertices: {vertices.shape,vertices.dtype}")
    #print(f"facets: {facets.shape,facets.dtype}")
    #print(f"normals: {normals.shape,normals.dtype}")
    
    numba_cuda.synchronize()
    thread_num = 256
    calculate_weighted_normals[(facets.shape[0]+thread_num-1)//thread_num,thread_num](normals_cuda, vertices_cuda, facets_cuda, facets_cuda.shape[0])
    normalize_normals[(normals_cuda.shape[0]+thread_num-1)//thread_num,thread_num](normals_cuda, normals_cuda.shape[0])
    numba_cuda.synchronize()
    normals = normals_cuda.copy_to_host()
    numba_cuda.synchronize()
    #print(f"normals 0: {normals[:10]}")
    return normals

def render_sequence(args, mm, torchutil, OSCRenderConfig):
    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    predicted_vertices_path = os.path.join(args.result_path,test_name+".npy")
    if args.dataset == "BIWI":
        template_file = os.path.join(args.dataset, args.render_template_path, "BIWI.ply")
    elif args.dataset == "vocaset":
        template_file = os.path.join(args.dataset, args.render_template_path, "FLAME_sample.ply")
         
    print("rendering: ", test_name)
                 
    with NVTXUtil("load Mesh", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        template = Mesh(filename=template_file)
        predicted_vertices = np.load(predicted_vertices_path)
        predicted_vertices = np.reshape(predicted_vertices,(-1,args.vertice_dim//3,3))

    output_path = args.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    tmp_nvc_video_file = tempfile.NamedTemporaryFile('w', suffix='_nvc.mp4', dir=output_path)
    print(f"tmp_video_file: {tmp_video_file}")
    print(f"tmp_nvc_video_file: {tmp_nvc_video_file}")
    
    with NVTXUtil("cv2.VideoWriter", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        center = np.mean(predicted_vertices[0], axis=0)
        t_center = center
        rot = np.zeros(3)
        with NVTXUtil("material", "red", mm):
            primitive_material = pyrender.material.MetallicRoughnessMaterial(
                        alphaMode='BLEND',
                        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
                        metallicFactor=0.8, 
                        roughnessFactor=0.8 
                    )

        renderer_grid_x = OSCRenderConfig["renderer_grid_x"]
        renderer_grid_y = OSCRenderConfig["renderer_grid_y"]
        frustum = OSCRenderConfig["frustum"]
        offscreen_renderer = OSCRenderConfig["offscreen_renderer"]

        render_meshes = []
        facets_cuda = None
        normals_cuda = numba_cuda.to_device(np.zeros(predicted_vertices[0].shape, dtype=np.float64))
        for i_frame in range(num_frames):
            with NVTXUtil("Mesh", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                mesh = Mesh(predicted_vertices[i_frame], template.f)
            
            with NVTXUtil("Rodrigues", "red", mm):
                mesh_copy = Mesh(mesh.v, mesh.f)
                mesh_copy.v[:] = cv2.Rodrigues(rot)[0].dot((mesh_copy.v-t_center).T).T+t_center
                intensity = 2.0
                rgb_per_v = None

            with NVTXUtil("trimesh", "red", mm):
                #print( f"tri_mesh = trimesh.Trimesh" )
                #print( f"mesh_copy.v: {mesh_copy.v.shape, mesh_copy.v.dtype}" )
                #print( f"mesh_copy.f: {mesh_copy.f.shape, mesh_copy.f.dtype}" )
                if True:
                    if facets_cuda is None:
                        facets_cuda = numba_cuda.to_device(mesh_copy.f)
                    normals = calculate_normals( normals_cuda, mesh_copy.v, mesh_copy.f, facets_cuda=facets_cuda )
                    #print(f"normals: {normals[:10]}")
                else:
                    normals = None
                tri_mesh = trimesh.Trimesh(vertices=mesh_copy.v, faces=mesh_copy.f, vertex_colors=rgb_per_v, vertex_normals=normals)
            with NVTXUtil("render_mesh", "red", mm):
                #print( f"render_mesh = pyrender.Mesh.from_trimesh" )
                render_mesh = pyrender.Mesh.from_trimesh(tri_mesh, material=primitive_material,smooth=True)

            render_meshes.append(render_mesh)
        
        if isinstance(offscreen_renderer, pyrender.OffscreenRendererCUDA):
            writer = None #cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
            nvc_surfaces = []
            for i_frame in range(num_frames):
                print(f"processing frame {i_frame}")
                with NVTXUtil(f"write{i_frame}", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                    #with NVTXUtil("Mesh", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                    #    render_mesh = Mesh(predicted_vertices[i_frame], template.f)
                    with NVTXUtil("render_mesh_helper_", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                        offset = i_frame%(renderer_grid_x*renderer_grid_y)
                        i_frame_offset_x = offset %  renderer_grid_x
                        i_frame_offset_y = offset // renderer_grid_x

                        if i_frame%(renderer_grid_x*renderer_grid_y)==0:
                            offscreen_renderer.render_init_grid_renderer( )
                            
                        scene = render_mesh_helper(args,render_meshes[i_frame], center, offscreen_renderer=offscreen_renderer, frustum=frustum, 
                        viewport_offset_x=i_frame_offset_x, viewport_offset_y=i_frame_offset_y,
                        mm=mm,
                        viewport_grid_x = renderer_grid_x,
                        viewport_grid_y = renderer_grid_y,)

                    with NVTXUtil("glFinish", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                        offscreen_renderer._renderer.glFinish( )

                    with NVTXUtil("get_rendered_output", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                        if (i_frame==num_frames-1) or (i_frame%(renderer_grid_x*renderer_grid_y)==(renderer_grid_x*renderer_grid_y-1)):
                            pred_img, nvc_surfs = render_mesh_helper_get_rendered_output_CUDA_OpenGL_Interop(  offscreen_renderer=offscreen_renderer, scene=scene )
                        else:
                            pred_img = None
                         
                    if pred_img is not None: 
                        if writer: pred_img = pred_img.astype(np.uint8)
                        #print(f"pred_img: {pred_img.shape, pred_img}")
                        i_frame_ = (i_frame // (renderer_grid_x*renderer_grid_y)) * (renderer_grid_x*renderer_grid_y)
                        for y in range(renderer_grid_y):
                            for x in range(renderer_grid_x):
                                if i_frame_<num_frames:
                                    if writer: pred_img_ = pred_img[   y*frustum["height"]:(y+1)*frustum["height"],
                                                            x*frustum["width"]:(x+1)*frustum["width"],
                                                            :]
                                    #cv2.imwrite( f"./demo/output/img_{i_frame_}.png", pred_img_ )
                                    with NVTXUtil("write", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                                        if writer: writer.write(pred_img_)
                                        #print(f"nvc_surfs: {len(nvc_surfs)}")
                                        nvc_surfaces.append(nvc_surfs[y*renderer_grid_x+x])
                                i_frame_ += 1
            
            print(f"fps: {args.fps}")
            CodecUtil.video_nvc_surfaces_to_VPF(nvc_surfaces, tmp_nvc_video_file.name, fps=args.fps, torch_stream=torchutil.torch_stream)
            if writer: writer.release()

            video_file = tmp_nvc_video_file
        elif isinstance(offscreen_renderer, pyrender.OffscreenRendererGrid):
            writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (800, 800), True)
            for i_frame in range(num_frames):
                print(f"processing frame {i_frame}")
                with NVTXUtil(f"write{i_frame}", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                    #with NVTXUtil("Mesh", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                    #    render_mesh = Mesh(predicted_vertices[i_frame], template.f)
                    with NVTXUtil("render_mesh_helper_", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                        offset = i_frame%(renderer_grid_x*renderer_grid_y)
                        i_frame_offset_x = offset %  renderer_grid_x
                        i_frame_offset_y = offset // renderer_grid_x

                        if i_frame%(renderer_grid_x*renderer_grid_y)==0:
                            offscreen_renderer.render_init_grid_renderer( )
                            
                        scene = render_mesh_helper(args,render_meshes[i_frame], center, offscreen_renderer=offscreen_renderer, frustum=frustum, 
                        viewport_offset_x=i_frame_offset_x, viewport_offset_y=i_frame_offset_y,
                        mm=mm,
                        viewport_grid_x = renderer_grid_x,
                        viewport_grid_y = renderer_grid_y,)

                    with NVTXUtil("glFinish", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                        offscreen_renderer._renderer.glFinish( )

                    with NVTXUtil("get_rendered_output", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                        if (i_frame==num_frames-1) or (i_frame%(renderer_grid_x*renderer_grid_y)==(renderer_grid_x*renderer_grid_y-1)):
                            pred_img, _ = render_mesh_helper_get_rendered_output_Grid(  offscreen_renderer=offscreen_renderer, scene=scene )
                        else:
                            pred_img = None
                                                
                    if pred_img is not None: 
                        if writer: pred_img = pred_img.astype(np.uint8)
                        i_frame_ = (i_frame // (renderer_grid_x*renderer_grid_y)) * (renderer_grid_x*renderer_grid_y)
                        for y in range(renderer_grid_y):
                            for x in range(renderer_grid_x):
                                if i_frame_<num_frames:
                                    if writer: pred_img_ = pred_img[   y*frustum["height"]:(y+1)*frustum["height"],
                                                            x*frustum["width"]:(x+1)*frustum["width"],
                                                            :]
                                    #cv2.imwrite( f"./demo/output/img_{i_frame_}.png", pred_img_ )
                                    with NVTXUtil("write", "red", mm): #, SynchronizeUtil(torchutil.torch_stream):
                                        if writer: writer.write(pred_img_)
                                i_frame_ += 1
            
            print(f"fps: {args.fps}")
            if writer: writer.release()
            video_file = tmp_video_file
        else:
            assert False

    file_name = test_name+"_"+args.subject+"_condition_"+args.condition

    video_fname = os.path.join(output_path, file_name+'.mp4')
    with NVTXUtil("ffmpeg", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        cmd = ('ffmpeg' + ' -y -i {0} -i {1} -map 0:v -map 1:a -c:v copy -c:a aac -pix_fmt yuv420p -qscale 0 {2}'.format(
        video_file.name, os.path.join(args.wav_path), video_fname)).split()
        call(cmd)

def get_offscreen_render_config():
    renderer_grid_x = 1
    renderer_grid_y = 6
    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}
    #offscreen_renderer = pyrender.OffscreenRendererGrid(viewport_width=frustum['width']*renderer_grid_x, viewport_height=frustum['height']*renderer_grid_y)
    offscreen_renderer = pyrender.OffscreenRendererCUDA(viewport_width=frustum['width']*renderer_grid_x, viewport_height=frustum['height']*renderer_grid_y)

    return {"renderer_grid_x": renderer_grid_x,
            "renderer_grid_y": renderer_grid_y,
            "frustum": frustum,
            "offscreen_renderer": offscreen_renderer,}

def main():
    import sys
    sys.path.insert(0, "/home/thor/projects/faceFormer/FaceFormer/CodecUtil.py")

    parser = argparse.ArgumentParser(description='FaceFormer: Speech-Driven 3D Facial Animation with Transformers')
    parser.add_argument("--model_name", type=str, default="biwi")
    parser.add_argument("--dataset", type=str, default="BIWI", help='vocaset or BIWI')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=25, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=23370*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F5 F6 F7 F8 M1 M2 M6")
    parser.add_argument("--output_path", type=str, default="demo/output", help='path of the rendered video sequence')
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav", help='path of the input audio signal')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')
    args = parser.parse_args()   

    OSCRenderConfig = get_offscreen_render_config()
    mm = Memory_Manager()
    mm.add_foot_print("prev-E2E")
    torchutil = TorchUtil(gpu=0, memory_manager=mm, cvcuda_stream=None)

    with NVTXUtil("test_model", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        test_model(args, mm, torchutil)
    
    with NVTXUtil("render_sequence", "red", mm), SynchronizeUtil(torchutil.torch_stream):
        render_sequence(args, mm, torchutil, OSCRenderConfig=OSCRenderConfig)

if __name__=="__main__":
    main()
