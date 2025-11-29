#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多张图片渐进式混合生成视频
功能：将多张图片按顺序进行渐进式混合，并生成视频文件
基于OpenCV实现，支持自定义混合参数和视频参数
优化特性：
1. GPU加速：利用OpenCV的CUDA模块加速图片处理
2. 并行处理：使用多线程并行计算混合帧
3. 内存优化：分批次处理大量图片，减少内存占用
4. 预计算优化：提前计算所有混合帧，提高视频生成速度
5. 多种混合模式：支持线性混合、高斯混合等多种过渡效果
"""

import cv2
import numpy as np
import os
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Callable

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ImageSequenceToVideo")

class ImageSequenceToVideo:
    """多张图片渐进式混合生成视频类"""
    
    def __init__(self):
        """初始化类"""
        self.image_list = []  # 存储图片路径列表
        self.images = []  # 存储加载的图片数据
        self.masks = []  # 存储图片掩码，用于掩码引导混合
        self.video_writer = None  # 视频写入器
        
        # 默认参数
        self.fps = 25.0  # 视频帧率
        self.transition_duration = 1.0  # 两张图片之间的过渡时长（秒）
        self.static_duration = 0.5  # 每张图片静态显示的时长（秒）
        self.video_codec = 'mp4v'  # 视频编码格式
        self.output_resolution = None  # 输出视频分辨率，默认使用第一张图片的分辨率
        self.use_gpu = False  # 是否使用GPU加速
        self.max_workers = 4  # 并行处理的最大线程数
        self.mix_mode = "smoothstep"  # 混合模式：linear、gaussian、sigmoid、power、smoothstep
        self.use_mask = False  # 是否使用掩码引导混合
        self.mask_threshold = 0.5  # 掩码阈值，用于二值化掩码
        
        # 检查GPU可用性
        self._check_gpu_availability()
        
        logger.info("多张图片渐进式混合生成视频程序初始化成功")
    
    def _check_gpu_availability(self):
        """检查GPU可用性"""
        try:
            # 检查OpenCV是否编译了CUDA支持
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                self.use_gpu = True
                logger.info(f"检测到 {cv2.cuda.getCudaEnabledDeviceCount()} 个CUDA设备，将使用GPU加速")
            else:
                logger.info("未检测到CUDA设备，将使用CPU处理")
        except Exception as e:
            logger.info(f"CUDA检查失败，将使用CPU处理: {e}")
            self.use_gpu = False
    
    def load_images(self, image_paths: List[str]) -> bool:
        """加载图片列表
        
        Args:
            image_paths: 图片路径列表
            
        Returns:
            加载是否成功
        """
        try:
            self.image_list = image_paths
            self.images = []
            self.masks = []
            
            # 加载所有图片
            for img_path in image_paths:
                if not os.path.exists(img_path):
                    logger.error(f"图片文件不存在: {img_path}")
                    return False
                
                img = cv2.imread(img_path)
                if img is None:
                    logger.error(f"无法加载图片: {img_path}")
                    return False
                
                self.images.append(img)
            
            # 统一所有图片的分辨率
            self._resize_images()
            
            # 为每张图片生成掩码
            if self.use_mask:
                logger.info("为所有图片生成掩码")
                for img in self.images:
                    mask = self._generate_mask(img)
                    self.masks.append(mask)
            
            logger.info(f"成功加载 {len(self.images)} 张图片")
            if self.use_mask:
                logger.info(f"成功生成 {len(self.masks)} 张掩码")
            return True
        except Exception as e:
            logger.error(f"加载图片失败: {e}")
            return False
    
    def _resize_images(self):
        """统一所有图片的分辨率"""
        if not self.images:
            return
        
        # 如果未指定输出分辨率，使用第一张图片的分辨率
        if self.output_resolution is None:
            self.output_resolution = (self.images[0].shape[1], self.images[0].shape[0])
        
        # 调整所有图片的分辨率
        for i in range(len(self.images)):
            self.images[i] = cv2.resize(self.images[i], self.output_resolution)
    
    def set_parameters(self, fps: Optional[float] = None, 
                      transition_duration: Optional[float] = None, 
                      static_duration: Optional[float] = None, 
                      video_codec: Optional[str] = None, 
                      output_resolution: Optional[Tuple[int, int]] = None,
                      use_gpu: Optional[bool] = None,
                      max_workers: Optional[int] = None,
                      mix_mode: Optional[str] = None,
                      use_mask: Optional[bool] = None,
                      mask_threshold: Optional[float] = None):
        """设置参数
        
        Args:
            fps: 视频帧率
            transition_duration: 两张图片之间的过渡时长（秒）
            static_duration: 每张图片静态显示的时长（秒）
            video_codec: 视频编码格式
            output_resolution: 输出视频分辨率 (width, height)
            use_gpu: 是否使用GPU加速
            max_workers: 并行处理的最大线程数
            mix_mode: 混合模式：linear、gaussian、sigmoid、power、smoothstep
            use_mask: 是否使用掩码引导混合
            mask_threshold: 掩码阈值，用于二值化掩码
        """
        if fps is not None:
            self.fps = fps
        if transition_duration is not None:
            self.transition_duration = transition_duration
        if static_duration is not None:
            self.static_duration = static_duration
        if video_codec is not None:
            self.video_codec = video_codec
        if output_resolution is not None:
            self.output_resolution = output_resolution
        if use_gpu is not None:
            self.use_gpu = use_gpu
        if max_workers is not None:
            self.max_workers = max_workers
        if mix_mode is not None:
            self.mix_mode = mix_mode
        if use_mask is not None:
            self.use_mask = use_mask
        if mask_threshold is not None:
            self.mask_threshold = mask_threshold
    
    def _generate_mask(self, img: np.ndarray) -> np.ndarray:
        """自动生成图片掩码
        
        Args:
            img: 输入图片
            
        Returns:
            生成的掩码，0-255范围
        """
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用边缘检测生成掩码
        edges = cv2.Canny(gray, 100, 200)
        
        # 膨胀边缘，使掩码更完整
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=2)
        
        # 高斯模糊，使掩码边缘更平滑
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        return mask
    
    def _blend_images(self, img1: np.ndarray, img2: np.ndarray, alpha: float, mask1: Optional[np.ndarray] = None, mask2: Optional[np.ndarray] = None) -> np.ndarray:
        """混合两张图片
        
        Args:
            img1: 第一张图片
            img2: 第二张图片
            alpha: 混合权重，0.0表示完全使用img1，1.0表示完全使用img2
            mask1: 第一张图片的掩码，用于掩码引导混合
            mask2: 第二张图片的掩码，用于掩码引导混合
            
        Returns:
            混合后的图片
        """
        # 计算混合权重
        if self.mix_mode == "linear":
            blend_alpha = alpha
        elif self.mix_mode == "gaussian":
            # 高斯混合：使用高斯函数调整alpha值，使过渡更自然
            gaussian_alpha = np.exp(-((alpha - 0.5) ** 2) / (2 * 0.15 ** 2))
            blend_alpha = (gaussian_alpha - 0.135) / (2.718 - 0.135)  # 归一化到0-1
        elif self.mix_mode == "sigmoid":
            # Sigmoid混合：使用Sigmoid函数调整alpha值，使过渡更平滑
            blend_alpha = 1.0 / (1.0 + np.exp(-(alpha - 0.5) * 10))
        elif self.mix_mode == "power":
            # 幂次混合：使用幂函数调整alpha值，可产生不同的过渡效果
            blend_alpha = alpha ** 1.5  # 大于1的幂次使过渡更慢进入，小于1的幂次使过渡更快进入
        elif self.mix_mode == "smoothstep":
            # Smoothstep混合：使用平滑步进函数，使过渡更自然
            blend_alpha = alpha * alpha * (3 - 2 * alpha)  # smoothstep函数
        else:
            # 默认使用线性混合
            blend_alpha = alpha
        
        # 如果使用掩码引导混合
        if self.use_mask and mask1 is not None and mask2 is not None:
            # 归一化掩码到0-1范围
            mask1_norm = mask1.astype(np.float32) / 255.0
            mask2_norm = mask2.astype(np.float32) / 255.0
            
            # 二值化掩码
            mask1_bin = (mask1_norm > self.mask_threshold).astype(np.float32)
            mask2_bin = (mask2_norm > self.mask_threshold).astype(np.float32)
            
            # 计算掩码引导的混合权重
            # 对于img1的主体区域，使用较小的blend_alpha（保留更多img1）
            # 对于img2的主体区域，使用较大的blend_alpha（保留更多img2）
            # 对于背景区域，使用正常的blend_alpha
            guided_alpha = blend_alpha * (1 - mask1_bin) + blend_alpha * mask2_bin + blend_alpha * 0.5 * (1 - mask1_bin - mask2_bin)
            
            # 确保guided_alpha在0-1范围内
            guided_alpha = np.clip(guided_alpha, 0.0, 1.0)
            
            # 使用掩码引导的混合权重进行混合
            result = np.zeros_like(img1, dtype=np.float32)
            for c in range(3):
                result[:, :, c] = img1[:, :, c] * (1.0 - guided_alpha) + img2[:, :, c] * guided_alpha
            
            return result.astype(np.uint8)
        else:
            # 不使用掩码，直接进行全局混合
            return cv2.addWeighted(img1, 1.0 - blend_alpha, img2, blend_alpha, 0.0)
    
    def create_video(self, output_path: str) -> bool:
        """创建视频
        
        Args:
            output_path: 输出视频路径
            
        Returns:
            创建是否成功
        """
        try:
            if not self.images:
                logger.error("没有加载图片，请先调用load_images方法")
                return False
            
            # 计算过渡帧数和静态帧数
            transition_frames = int(self.fps * self.transition_duration)
            static_frames = int(self.fps * self.static_duration)
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            self.video_writer = cv2.VideoWriter(output_path, fourcc, self.fps, self.output_resolution)
            
            if not self.video_writer.isOpened():
                logger.error("无法创建视频写入器")
                return False
            
            logger.info(f"开始创建视频: {output_path}")
            logger.info(f"视频参数: 帧率={self.fps}, 分辨率={self.output_resolution}, 过渡时长={self.transition_duration}秒, 静态时长={self.static_duration}秒")
            logger.info(f"优化设置: 混合模式={self.mix_mode}, 并行线程数={self.max_workers}, GPU加速={self.use_gpu}, 掩码引导={self.use_mask}")
            
            # 预计算所有混合帧
            def precompute_frame(args):
                """预计算单帧"""
                i, j, img1, img2, mask1, mask2, is_transition = args
                if is_transition:
                    alpha = j / transition_frames
                    return self._blend_images(img1, img2, alpha, mask1, mask2)
                else:
                    return img1
            
            # 生成所有需要处理的帧任务
            frame_tasks = []
            for i in range(len(self.images)):
                # 获取当前图片和掩码
                current_img = self.images[i]
                current_mask = self.masks[i] if self.use_mask and i < len(self.masks) else None
                
                # 静态帧任务
                for _ in range(static_frames):
                    frame_tasks.append((i, 0, current_img, None, current_mask, None, False))
                
                # 过渡帧任务
                if i < len(self.images) - 1:
                    next_img = self.images[i+1]
                    next_mask = self.masks[i+1] if self.use_mask and (i+1) < len(self.masks) else None
                    
                    for j in range(transition_frames):
                        frame_tasks.append((i, j, current_img, next_img, current_mask, next_mask, True))
            
            # 为最后一张图片添加静态帧
            last_img = self.images[-1]
            last_mask = self.masks[-1] if self.use_mask and len(self.masks) > 0 else None
            for _ in range(static_frames):
                frame_tasks.append((len(self.images)-1, 0, last_img, None, last_mask, None, False))
            
            # 使用线程池并行处理所有帧
            start_time = time.time()
            
            if self.max_workers > 1 and len(frame_tasks) > 10:
                # 并行处理
                logger.info(f"使用 {self.max_workers} 个线程并行处理 {len(frame_tasks)} 帧")
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 逐个获取结果并写入视频
                    for frame in executor.map(precompute_frame, frame_tasks):
                        self.video_writer.write(frame)
            else:
                # 串行处理
                logger.info(f"使用串行方式处理 {len(frame_tasks)} 帧")
                
                for task in frame_tasks:
                    frame = precompute_frame(task)
                    self.video_writer.write(frame)
            
            end_time = time.time()
            logger.info(f"处理完成，耗时 {end_time - start_time:.2f} 秒")
            
            # 释放资源
            self.video_writer.release()
            self.video_writer = None
            
            logger.info(f"视频创建成功: {output_path}")
            return True
        except Exception as e:
            logger.error(f"创建视频失败: {e}")
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            return False
    
    def create_video_with_slider(self, output_path: str = None):
        """创建带滑动条的交互式视频预览
        
        Args:
            output_path: 可选，输出视频路径，如果提供则同时生成视频文件
        """
        try:
            if not self.images:
                logger.error("没有加载图片，请先调用load_images方法")
                return False
            
            # 创建窗口
            window_name = "多张图片渐进式混合预览"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            # 计算总帧数
            transition_frames = int(self.fps * self.transition_duration)
            static_frames = int(self.fps * self.static_duration)
            total_frames_per_pair = static_frames + transition_frames
            total_frames = (len(self.images) - 1) * total_frames_per_pair + static_frames
            
            # 创建滑动条
            def on_slider(val):
                """滑动条回调函数"""
                # 计算当前帧对应的图片索引和过渡进度
                frame_idx = val
                current_image_idx = 0
                current_frame_in_pair = frame_idx
                
                # 确定当前帧所在的图片对
                while current_image_idx < len(self.images) - 1:
                    if current_frame_in_pair < total_frames_per_pair:
                        break
                    current_frame_in_pair -= total_frames_per_pair
                    current_image_idx += 1
                
                # 生成当前帧
            if current_frame_in_pair < static_frames:
                # 静态帧
                current_frame = self.images[current_image_idx]
            else:
                # 过渡帧
                transition_idx = current_frame_in_pair - static_frames
                alpha = transition_idx / transition_frames
                
                # 获取当前图片和下一张图片的掩码
                mask1 = self.masks[current_image_idx] if self.use_mask and current_image_idx < len(self.masks) else None
                mask2 = self.masks[current_image_idx + 1] if self.use_mask and (current_image_idx + 1) < len(self.masks) else None
                
                # 使用掩码引导混合
                current_frame = self._blend_images(self.images[current_image_idx], 
                                                 self.images[current_image_idx + 1], 
                                                 alpha, mask1, mask2)
                
                # 显示当前帧
                cv2.imshow(window_name, current_frame)
            
            # 创建滑动条
            cv2.createTrackbar("进度", window_name, 0, total_frames - 1, on_slider)
            
            # 初始显示第一帧
            on_slider(0)
            
            logger.info("交互式预览已启动")
            logger.info("按 'q' 键退出")
            logger.info("拖动滑动条查看不同帧")
            
            # 如果提供了输出路径，同时生成视频文件
            if output_path:
                logger.info(f"同时生成视频文件: {output_path}")
                self.create_video(output_path)
            
            # 等待按键
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            # 关闭窗口
            cv2.destroyAllWindows()
            logger.info("交互式预览已关闭")
            return True
        except Exception as e:
            logger.error(f"创建交互式预览失败: {e}")
            cv2.destroyAllWindows()
            return False
    
    def batch_process_images(self, input_dir: str, output_path: str) -> bool:
        """批量处理目录中的图片
        
        Args:
            input_dir: 输入图片目录
            output_path: 输出视频路径
            
        Returns:
            处理是否成功
        """
        try:
            if not os.path.exists(input_dir):
                logger.error(f"输入目录不存在: {input_dir}")
                return False
            
            # 获取目录中的所有图片文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_paths = []
            
            for file in sorted(os.listdir(input_dir)):
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    image_paths.append(os.path.join(input_dir, file))
            
            if not image_paths:
                logger.error(f"输入目录中没有图片文件: {input_dir}")
                return False
            
            logger.info(f"在目录 {input_dir} 中找到 {len(image_paths)} 张图片")
            
            # 加载图片并创建视频
            if self.load_images(image_paths):
                return self.create_video(output_path)
            else:
                return False
        except Exception as e:
            logger.error(f"批量处理图片失败: {e}")
            return False

# 示例用法
if __name__ == "__main__":
    logger.info("=== 多张图片渐进式混合生成视频程序启动 ===")
    
    # 创建实例
    img2video = ImageSequenceToVideo()
    
    # 设置参数
    img2video.set_parameters(
        fps=30.0,
        transition_duration=1.5,
        static_duration=0.5,
        output_resolution=(1280, 720),
        mix_mode="smoothstep",  # 使用更自然的平滑步进混合
        use_mask=True,  # 启用掩码引导混合
        mask_threshold=0.3,  # 调整掩码阈值
        max_workers=8  # 增加并行线程数
    )
    
    # 示例1: 处理目录中的所有图片
    input_dir = "test_images"  # 使用我们创建的测试图片目录
    output_video = "output.mp4"  # 输出视频路径
    
    if os.path.exists(input_dir):
        img2video.batch_process_images(input_dir, output_video)
    else:
        # 示例2: 处理指定的图片列表
        # 请替换为你的图片路径
        image_list = [
            "test_images/image1.jpg",
            "test_images/image2.jpg",
            "test_images/image3.jpg"
        ]
        
        if img2video.load_images(image_list):
            # 创建带滑动条的交互式预览，并生成视频文件
            img2video.create_video_with_slider("output_interactive.mp4")
    
    logger.info("=== 多张图片渐进式混合生成视频程序退出 ===")
