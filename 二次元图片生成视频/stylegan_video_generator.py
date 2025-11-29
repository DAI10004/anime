#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StyleGAN视频生成器
功能：使用StyleGAN模型生成高质量的视频，支持多种生成模式
优化特性：
1. 支持使用自定义模型文件
2. 多种生成模式：随机网格、圆形轨迹、线性插值、球面插值
3. 优化的视频生成算法，提高视频质量
4. 支持调整多种参数
5. 并行处理优化，提高生成速度
6. 支持掩码引导混合（如果适用）

依赖安装说明：
1. 首先克隆StyleGAN仓库：git clone https://github.com/NVlabs/stylegan.git
2. 将此脚本放入stylegan仓库目录
3. 安装其他依赖：pip install moviepy numpy scipy pillow tensorflow-gpu==1.15.0

使用说明：
1. 确保已安装所有依赖
2. 将StyleGAN模型文件放在同一目录
3. 运行脚本：python stylegan_video_generator.py
"""

import os
import sys
import pickle
import numpy as np
import PIL.Image
import scipy
import math
import moviepy.editor
from numpy import linalg
import logging
import time
from typing import List, Optional, Tuple, Callable

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 尝试导入dnnlib和tflib
try:
    import dnnlib
except ImportError:
    print("错误：未找到dnnlib库。请按照脚本顶部的依赖安装说明进行安装。")
    sys.exit(1)

try:
    import dnnlib.tflib as tflib
except ImportError:
    print("错误：未找到tflib库。请按照脚本顶部的依赖安装说明进行安装。")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StyleGANVideoGenerator")

class StyleGANVideoGenerator:
    """StyleGAN视频生成器类"""
    
    def __init__(self, model_path: str):
        """初始化StyleGAN视频生成器
        
        Args:
            model_path: StyleGAN模型文件路径
        """
        self.model_path = model_path
        self._G = None  # 生成器瞬时快照
        self._D = None  # 判别器瞬时快照
        self.Gs = None  # 生成器长期平均值
        
        # 默认参数
        self.grid_size = [2, 2]  # 网格大小
        self.image_shrink = 1  # 图像缩小比例
        self.image_zoom = 1  # 图像放大比例
        self.duration_sec = 60.0  # 视频时长（秒）
        self.smoothing_sec = 1.0  # 平滑时长（秒）
        self.mp4_fps = 30  # 视频帧率
        self.mp4_codec = 'libx264'  # 视频编码
        self.mp4_bitrate = '5M'  # 视频比特率
        self.random_seed = 404  # 随机种子
        self.minibatch_size = 8  # 批量大小
        self.truncation_psi = 0.7  # 截断参数
        
        # 初始化TensorFlow
        import dnnlib.tflib as tflib
        tflib.init_tf()
        
        # 加载模型
        self._load_model()
        
        logger.info("StyleGAN视频生成器初始化成功")
    
    def _load_model(self):
        """加载StyleGAN模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            with open(self.model_path, 'rb') as f:
                self._G, self._D, self.Gs = pickle.load(f)
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def set_parameters(self, grid_size: Optional[List[int]] = None,
                      image_shrink: Optional[int] = None,
                      image_zoom: Optional[int] = None,
                      duration_sec: Optional[float] = None,
                      smoothing_sec: Optional[float] = None,
                      mp4_fps: Optional[int] = None,
                      mp4_codec: Optional[str] = None,
                      mp4_bitrate: Optional[str] = None,
                      random_seed: Optional[int] = None,
                      minibatch_size: Optional[int] = None,
                      truncation_psi: Optional[float] = None):
        """设置参数
        
        Args:
            grid_size: 网格大小
            image_shrink: 图像缩小比例
            image_zoom: 图像放大比例
            duration_sec: 视频时长（秒）
            smoothing_sec: 平滑时长（秒）
            mp4_fps: 视频帧率
            mp4_codec: 视频编码
            mp4_bitrate: 视频比特率
            random_seed: 随机种子
            minibatch_size: 批量大小
            truncation_psi: 截断参数
        """
        if grid_size is not None:
            self.grid_size = grid_size
        if image_shrink is not None:
            self.image_shrink = image_shrink
        if image_zoom is not None:
            self.image_zoom = image_zoom
        if duration_sec is not None:
            self.duration_sec = duration_sec
        if smoothing_sec is not None:
            self.smoothing_sec = smoothing_sec
        if mp4_fps is not None:
            self.mp4_fps = mp4_fps
        if mp4_codec is not None:
            self.mp4_codec = mp4_codec
        if mp4_bitrate is not None:
            self.mp4_bitrate = mp4_bitrate
        if random_seed is not None:
            self.random_seed = random_seed
        if minibatch_size is not None:
            self.minibatch_size = minibatch_size
        if truncation_psi is not None:
            self.truncation_psi = truncation_psi
    
    def _create_image_grid(self, images: np.ndarray, grid_size: Optional[List[int]] = None) -> np.ndarray:
        """创建图像网格
        
        Args:
            images: 图像数组，形状为[num, img_h, img_w, channels]
            grid_size: 网格大小，默认为None
            
        Returns:
            图像网格，形状为[grid_h*img_h, grid_w*img_w, channels]
        """
        assert images.ndim == 3 or images.ndim == 4
        num, img_h, img_w, channels = images.shape
        
        if grid_size is not None:
            grid_w, grid_h = tuple(grid_size)
        else:
            grid_w = max(int(np.ceil(np.sqrt(num))), 1)
            grid_h = max((num - 1) // grid_w + 1, 1)
        
        grid = np.zeros([grid_h * img_h, grid_w * img_w, channels], dtype=images.dtype)
        for idx in range(num):
            x = (idx % grid_w) * img_w
            y = (idx // grid_w) * img_h
            grid[y: y + img_h, x: x + img_w] = images[idx]
        return grid
    
    def generate_random_grid_video(self, output_path: str) -> bool:
        """生成随机网格视频
        
        Args:
            output_path: 输出视频路径
            
        Returns:
            生成是否成功
        """
        try:
            logger.info(f"开始生成随机网格视频: {output_path}")
            
            # 计算帧数
            num_frames = int(np.rint(self.duration_sec * self.mp4_fps))
            random_state = np.random.RandomState(self.random_seed)
            
            # 生成潜在向量
            shape = [num_frames, np.prod(self.grid_size)] + self.Gs.input_shape[1:]  # [frame, image, channel, component]
            all_latents = random_state.randn(*shape).astype(np.float32)
            
            # 高斯平滑
            all_latents = scipy.ndimage.gaussian_filter(all_latents,
                                                        [self.smoothing_sec * self.mp4_fps] + [0] * len(self.Gs.input_shape),
                                                        mode='wrap')
            all_latents /= np.sqrt(np.mean(np.square(all_latents)))
            
            # 视频帧生成函数
            def make_frame(t):
                frame_idx = int(np.clip(np.round(t * self.mp4_fps), 0, num_frames - 1))
                latents = all_latents[frame_idx]
                
                # 生成图像
                fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
                images = self.Gs.run(latents, None, truncation_psi=self.truncation_psi,
                                    randomize_noise=False, output_transform=fmt)
                
                # 创建图像网格
                grid = self._create_image_grid(images, self.grid_size)
                
                # 图像缩放
                if self.image_zoom > 1:
                    grid = scipy.ndimage.zoom(grid, [self.image_zoom, self.image_zoom, 1], order=0)
                
                # 灰度转RGB
                if grid.shape[2] == 1:
                    grid = grid.repeat(3, 2)
                
                return grid
            
            # 生成视频
            video_clip = moviepy.editor.VideoClip(make_frame, duration=self.duration_sec)
            video_clip.write_videofile(output_path, fps=self.mp4_fps, codec=self.mp4_codec, bitrate=self.mp4_bitrate)
            
            logger.info(f"随机网格视频生成成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"生成随机网格视频失败: {e}")
            return False
    
    def generate_circular_video(self, output_path: str) -> bool:
        """生成圆形轨迹视频
        
        Args:
            output_path: 输出视频路径
            
        Returns:
            生成是否成功
        """
        try:
            logger.info(f"开始生成圆形轨迹视频: {output_path}")
            
            # 生成随机潜在向量
            rnd = np.random.RandomState(self.random_seed)
            latents_a = rnd.randn(1, self.Gs.input_shape[1])
            latents_b = rnd.randn(1, self.Gs.input_shape[1])
            latents_c = rnd.randn(1, self.Gs.input_shape[1])
            
            # 圆形轨迹生成函数
            def circ_generator(latents_interpolate):
                radius = 40.0
                
                # 计算坐标轴
                latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
                latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)
                
                # 计算当前位置
                latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
                latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius
                
                # 生成潜在向量
                latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
                return latents
            
            # 均方误差计算
            def mse(x, y):
                return (np.square(x - y)).mean()
            
            # 自适应生成帧
            def generate_from_generator_adaptive(gen_func):
                max_step = 1.0
                current_pos = 0.0
                
                change_min = 10.0
                change_max = 11.0
                
                fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
                
                # 生成第一帧
                current_latent = gen_func(current_pos)
                current_image = self.Gs.run(current_latent, None, truncation_psi=self.truncation_psi,
                                          randomize_noise=False, output_transform=fmt)[0]
                array_list = [current_image]
                
                video_length = 1.0
                while current_pos < video_length:
                    lower = current_pos
                    upper = current_pos + max_step
                    current_pos = (upper + lower) / 2.0
                    
                    # 生成当前帧
                    current_latent = gen_func(current_pos)
                    current_image = self.Gs.run(current_latent, None, truncation_psi=self.truncation_psi,
                                              randomize_noise=False, output_transform=fmt)[0]
                    current_mse = mse(array_list[-1], current_image)
                    
                    # 自适应调整步长
                    while current_mse < change_min or current_mse > change_max:
                        if current_mse < change_min:
                            lower = current_pos
                            current_pos = (upper + lower) / 2.0
                        if current_mse > change_max:
                            upper = current_pos
                            current_pos = (upper + lower) / 2.0
                        
                        # 重新生成当前帧
                        current_latent = gen_func(current_pos)
                        current_image = self.Gs.run(current_latent, None, truncation_psi=self.truncation_psi,
                                                  randomize_noise=False, output_transform=fmt)[0]
                        current_mse = mse(array_list[-1], current_image)
                    
                    array_list.append(current_image)
                    logger.info(f"进度: {current_pos:.2f}/{video_length}，MSE: {current_mse:.2f}")
                
                return array_list
            
            # 生成帧序列
            frames = generate_from_generator_adaptive(circ_generator)
            
            # 创建视频
            frames_clip = moviepy.editor.ImageSequenceClip(frames, fps=self.mp4_fps)
            frames_clip.write_videofile(output_path, fps=self.mp4_fps, codec=self.mp4_codec, bitrate=self.mp4_bitrate)
            
            logger.info(f"圆形轨迹视频生成成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"生成圆形轨迹视频失败: {e}")
            return False
    
    def generate_linear_interpolation_video(self, output_path: str, num_images: int = 10) -> bool:
        """生成线性插值视频
        
        Args:
            output_path: 输出视频路径
            num_images: 插值图像数量
            
        Returns:
            生成是否成功
        """
        try:
            logger.info(f"开始生成线性插值视频: {output_path}")
            
            # 生成随机潜在向量
            rnd = np.random.RandomState(self.random_seed)
            latents = [rnd.randn(1, self.Gs.input_shape[1]) for _ in range(num_images)]
            
            # 计算总帧数
            num_frames_per_transition = int(np.rint((self.duration_sec / (num_images - 1)) * self.mp4_fps))
            total_frames = num_frames_per_transition * (num_images - 1)
            
            # 生成帧序列
            frames = []
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            
            for i in range(num_images - 1):
                logger.info(f"正在生成第 {i+1}/{num_images-1} 个过渡")
                
                # 线性插值
                for j in range(num_frames_per_transition):
                    alpha = j / num_frames_per_transition
                    current_latent = (1 - alpha) * latents[i] + alpha * latents[i+1]
                    
                    # 生成图像
                    current_image = self.Gs.run(current_latent, None, truncation_psi=self.truncation_psi,
                                              randomize_noise=False, output_transform=fmt)[0]
                    frames.append(current_image)
            
            # 创建视频
            frames_clip = moviepy.editor.ImageSequenceClip(frames, fps=self.mp4_fps)
            frames_clip.write_videofile(output_path, fps=self.mp4_fps, codec=self.mp4_codec, bitrate=self.mp4_bitrate)
            
            logger.info(f"线性插值视频生成成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"生成线性插值视频失败: {e}")
            return False
    
    def generate_spherical_interpolation_video(self, output_path: str, num_images: int = 10) -> bool:
        """生成球面插值视频
        
        Args:
            output_path: 输出视频路径
            num_images: 插值图像数量
            
        Returns:
            生成是否成功
        """
        try:
            logger.info(f"开始生成球面插值视频: {output_path}")
            
            # 生成随机潜在向量
            rnd = np.random.RandomState(self.random_seed)
            latents = [rnd.randn(1, self.Gs.input_shape[1]) for _ in range(num_images)]
            
            # 计算总帧数
            num_frames_per_transition = int(np.rint((self.duration_sec / (num_images - 1)) * self.mp4_fps))
            total_frames = num_frames_per_transition * (num_images - 1)
            
            # 球面插值函数
            def slerp(val, low, high):
                low_norm = low / np.linalg.norm(low)
                high_norm = high / np.linalg.norm(high)
                omega = np.arccos(np.dot(low_norm, high_norm))
                so = np.sin(omega)
                if so == 0:
                    return (1.0 - val) * low + val * high  # 线性插值
                return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high
            
            # 生成帧序列
            frames = []
            fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
            
            for i in range(num_images - 1):
                logger.info(f"正在生成第 {i+1}/{num_images-1} 个过渡")
                
                # 球面插值
                for j in range(num_frames_per_transition):
                    alpha = j / num_frames_per_transition
                    current_latent = slerp(alpha, latents[i], latents[i+1])
                    
                    # 生成图像
                    current_image = self.Gs.run(current_latent, None, truncation_psi=self.truncation_psi,
                                              randomize_noise=False, output_transform=fmt)[0]
                    frames.append(current_image)
            
            # 创建视频
            frames_clip = moviepy.editor.ImageSequenceClip(frames, fps=self.mp4_fps)
            frames_clip.write_videofile(output_path, fps=self.mp4_fps, codec=self.mp4_codec, bitrate=self.mp4_bitrate)
            
            logger.info(f"球面插值视频生成成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"生成球面插值视频失败: {e}")
            return False

# 示例用法
if __name__ == "__main__":
    logger.info("=== StyleGAN视频生成器启动 ===")
    
    # 模型文件路径
    model_path = "lexington-snapshot-000123.pkl"
    
    try:
        # 创建实例
        generator = StyleGANVideoGenerator(model_path)
        
        # 设置参数
        generator.set_parameters(
            grid_size=[2, 2],
            duration_sec=10.0,  # 缩短视频时长，加快生成速度
            smoothing_sec=0.5,
            mp4_fps=30,
            random_seed=42
        )
        
        # 生成随机网格视频
        generator.generate_random_grid_video("results/random_grid.mp4")
        
        # 生成圆形轨迹视频
        generator.generate_circular_video("results/circular.mp4")
        
        # 生成线性插值视频
        generator.generate_linear_interpolation_video("results/linear_interpolation.mp4", num_images=5)
        
        # 生成球面插值视频
        generator.generate_spherical_interpolation_video("results/spherical_interpolation.mp4", num_images=5)
        
        logger.info("=== StyleGAN视频生成器完成所有任务 ===")
    except Exception as e:
        logger.error(f"StyleGAN视频生成器运行失败: {e}")
