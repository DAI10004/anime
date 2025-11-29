#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
人脸实时动漫化程序
功能：通过摄像头识别用户人脸，使用 AnimeGANv2 将人脸实时动漫化
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
from typing import Tuple, Optional
from mtcnn import MTCNN

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceToAnime")

# 定义 AnimeGANv2 模型结构（与权重文件完全匹配）
class ConvNormLReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):
        pad_layer = {
            "zero":    nn.ZeroPad2d,
            "same":    nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError
            
        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=0, groups=groups, bias=bias),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )

class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch*expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))
        
        # dw
        layers.append(ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True))
        # pw
        layers.append(nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False))
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out

class AnimeGANV2(nn.Module):
    def __init__(self, ):        
        super().__init__()
        
        self.block_a = nn.Sequential(
            ConvNormLReLU(3,  32, kernel_size=7, padding=3),
            ConvNormLReLU(32, 64, stride=2, padding=(0,1,0,1)),
            ConvNormLReLU(64, 64)
        )
        
        self.block_b = nn.Sequential(
            ConvNormLReLU(64,  128, stride=2, padding=(0,1,0,1)),            
            ConvNormLReLU(128, 128)
        )
        
        self.block_c = nn.Sequential(
            ConvNormLReLU(128, 128),
            InvertedResBlock(128, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            InvertedResBlock(256, 256, 2),
            ConvNormLReLU(256, 128),
        )    
        
        self.block_d = nn.Sequential(
            ConvNormLReLU(128, 128),
            ConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            ConvNormLReLU(128, 64),
            ConvNormLReLU(64,  64),
            ConvNormLReLU(64,  32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )
        
    def forward(self, input, align_corners=True):
        out = self.block_a(input)
        half_size = out.size()[-2:]
        out = self.block_b(out)
        out = self.block_c(out)
        
        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
        out = self.block_e(out)

        out = self.out_layer(out)
        return out

class FaceToAnime:
    """人脸实时动漫化类"""
    
    def __init__(self, weight_path: str):
        """初始化人脸实时动漫化类
        
        Args:
            weight_path: AnimeGANv2 权重文件路径
        """
        # 初始化状态
        self.running = False
        self.camera = None
        
        # 初始化 MTCNN 人脸检测器（用于更精确的人脸检测）
        self.mtcnn = MTCNN(device='cpu')
        
        # 预训练权重列表，支持多种动漫风格
        self.weight_list = [
            "../weights/face_paint_512_v2.pt",  # 默认风格
            "../weights/face_paint_512_v1.pt",  # 风格1
            "../weights/celeba_distill.pt",      # 风格2
            "../weights/paprika.pt"              # 风格3
        ]
        
        # 当前权重索引
        self.current_weight_idx = 0
        
        # 加载 AnimeGANv2 模型
        self._load_animegan_model(weight_path)
        
        # 默认输入分辨率为512x512，可通过键盘快捷键调整
        self.input_resolution = 512
        
        # 视频录制相关
        self.is_recording = False
        self.video_writer = None
        
        logger.info("人脸实时动漫化程序初始化成功")
    
    def _load_animegan_model(self, weight_path: str):
        """加载 AnimeGANv2 模型
        
        Args:
            weight_path: 权重文件路径
        """
        try:
            # 加载权重文件
            weights = torch.load(weight_path, map_location='cpu')
            
            # 打印权重文件的键名，帮助调试
            logger.info(f"权重文件包含 {len(weights)} 个键")
            for i, key in enumerate(list(weights.keys())[:10]):
                logger.info(f"权重键名 {i+1}: {key}")
            
            # 初始化完整的 AnimeGANv2 模型
            self.animegan = AnimeGANV2()
            
            # 加载权重，使用strict=True确保完全匹配
            self.animegan.load_state_dict(weights, strict=True)
            logger.info("成功加载所有权重")
            
            self.animegan.eval()
            
            # 检查 GPU 可用性
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.animegan.to(self.device)
            
            logger.info(f"成功初始化 AnimeGANv2 模型")
        except Exception as e:
            logger.error(f"加载 AnimeGANv2 模型失败: {e}")
            raise
    
    def init_camera(self, camera_id: int = 0, width: int = 1280, height: int = 720) -> bool:
        """初始化摄像头
        
        Args:
            camera_id: 摄像头 ID
            width: 摄像头宽度，默认1280以兼容用户相机
            height: 摄像头高度，默认720以兼容用户相机
            
        Returns:
            初始化是否成功
        """
        try:
            # 打开摄像头
            self.camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)  # 使用 DSHOW 后端避免 Windows 摄像头问题
            if not self.camera.isOpened():
                logger.error(f"无法打开摄像头 {camera_id}")
                return False
            
            # 设置摄像头分辨率为1280x720，兼容用户相机
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 获取实际设置的分辨率
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            logger.info(f"摄像头初始化成功: 设置分辨率 {width}x{height}，实际分辨率 {actual_width}x{actual_height}")
            return True
        except Exception as e:
            logger.error(f"初始化摄像头失败: {e}")
            return False
    
    def _get_face_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """使用 MTCNN 检测人脸边界框
        
        Args:
            frame: 输入图像帧
            
        Returns:
            人脸边界框，格式为 (x, y, w, h)，如果没有检测到人脸则返回 None
        """
        try:
            # MTCNN 检测人脸，返回边界框坐标
            boxes = self.mtcnn.detect_faces(frame)
            
            if not boxes:
                return None
            
            # 获取第一个检测到的人脸边界框
            box = boxes[0]['box']
            x, y, w, h = box
            
            # 确保边界框有效
            if w <= 0 or h <= 0:
                return None
            
            return (x, y, w, h)
        except Exception as e:
            logger.error(f"检测人脸失败: {e}")
            return None
    
    def _animegan_process(self, face_region: np.ndarray) -> np.ndarray:
        """使用 AnimeGANv2 处理人脸区域
        
        Args:
            face_region: 人脸区域图像
            
        Returns:
            动漫化后的人脸图像
        """
        try:
            # 预处理：调整大小、归一化、转换为张量
            # 使用可调整的输入分辨率，降低分辨率可提高处理速度
            face_resized = cv2.resize(face_region, (self.input_resolution, self.input_resolution))
            face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).float() / 255.0
            face_tensor = (face_tensor - 0.5) * 2.0  # 归一化到 [-1, 1]
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            
            # 推理
            with torch.no_grad():
                anime_tensor = self.animegan(face_tensor)
            
            # 后处理：转换回 numpy 数组
            anime_tensor = (anime_tensor + 1.0) / 2.0  # 归一化到 [0, 1]
            anime_np = anime_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            anime_np = (anime_np * 255).astype(np.uint8)
            
            # 调整回原始人脸区域大小
            anime_face = cv2.resize(anime_np, (face_region.shape[1], face_region.shape[0]))
            
            return anime_face
        except Exception as e:
            logger.error(f"AnimeGANv2 处理失败: {e}")
            return face_region
    
    def _blend_anime_face(self, frame: np.ndarray, anime_face: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """将动漫化后的人脸融合回原图
        
        使用泊松融合算法，实现更自然的人脸融合效果
        
        Args:
            frame: 原始图像帧
            anime_face: 动漫化后的人脸图像
            position: 人脸在原始图像中的位置 (x, y)
            
        Returns:
            融合后的图像帧
        """
        x, y = position
        h, w = anime_face.shape[:2]
        
        # 确保融合区域在图像范围内
        y_end = min(y + h, frame.shape[0])
        x_end = min(x + w, frame.shape[1])
        h_crop = y_end - y
        w_crop = x_end - x
        
        # 裁剪动漫人脸到实际可用大小
        anime_face_crop = anime_face[:h_crop, :w_crop]
        
        # 创建融合区域的掩码
        mask = np.ones((h_crop, w_crop), dtype=np.uint8) * 255
        
        # 边缘模糊处理，使融合更自然
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # 创建ROI区域
        roi = frame[y:y_end, x:x_end]
        
        try:
            # 使用泊松融合（seamlessClone）实现更自然的融合效果
            # 中心坐标用于泊松融合
            center = (w_crop // 2, h_crop // 2)
            
            # 泊松融合：参数说明
            # src: 源图像（动漫人脸）
            # dst: 目标图像（原始帧的ROI）
            # mask: 掩码，白色区域表示需要融合的部分
            # center: 源图像在目标图像中的中心位置
            # flags: 融合方式，NORMAL_CLONE表示普通克隆
            blended_face = cv2.seamlessClone(anime_face_crop, roi, mask, center, cv2.NORMAL_CLONE)
            
            # 将融合后的人脸放回原图
            frame[y:y_end, x:x_end] = blended_face
        except Exception as e:
            logger.error(f"泊松融合失败，使用简单融合: {e}")
            # 备用方案：使用简单的加权融合
            mask_float = mask.astype(np.float32) / 255.0
            frame[y:y_end, x:x_end] = (roi * (1 - mask_float[..., np.newaxis]) + 
                                      anime_face_crop * mask_float[..., np.newaxis]).astype(np.uint8)
        
        return frame
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """处理单帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            处理后的图像帧
        """
        if not self.running:
            return frame
        
        # 水平翻转帧（镜像效果）
        frame = cv2.flip(frame, 1)
        
        # 获取人脸边界框
        face_bbox = self._get_face_bbox(frame)
        
        if face_bbox is not None:
            x, y, w, h = face_bbox
            
            # 扩展人脸区域，获得更好的动漫化效果
            expand_ratio = 0.3
            x_expand = max(0, int(x - w * expand_ratio))
            y_expand = max(0, int(y - h * expand_ratio))
            w_expand = int(w * (1 + 2 * expand_ratio))
            h_expand = int(h * (1 + 2 * expand_ratio))
            
            # 提取扩展后的人脸区域
            face_region = frame[y_expand:y_expand+h_expand, x_expand:x_expand+w_expand]
            
            # 动漫化处理
            anime_face = self._animegan_process(face_region)
            
            # 将动漫化后的人脸融合回原图
            frame = self._blend_anime_face(frame, anime_face, (x_expand, y_expand))
        
        return frame
    
    def run(self) -> bool:
        """运行程序
        
        Returns:
            运行是否成功
        """
        # 直接设置 running 为 True
        self.running = True
        
        if not self.init_camera():
            self.shutdown()
            return False
        
        logger.info("人脸实时动漫化程序开始运行")
        logger.info("按 'q' 键退出")
        logger.info("按 's' 键切换风格")
        logger.info("按 '+' 键提高分辨率")
        logger.info("按 '-' 键降低分辨率")
        logger.info("按 'r' 键开始/停止录制视频")
        
        try:
            while self.running:
                # 读取摄像头帧
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("无法读取摄像头帧")
                    time.sleep(0.1)
                    continue
                
                # 处理帧
                processed_frame = self.process_frame(frame)
                
                # 显示当前状态信息
                info_text = f"风格: {self.current_weight_idx+1}/{len(self.weight_list)} | 分辨率: {self.input_resolution}x{self.input_resolution} | 录制: {'开启' if self.is_recording else '关闭'}"
                cv2.putText(processed_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow("Face to Anime", processed_frame)
                
                # 视频录制
                if self.is_recording and self.video_writer is not None:
                    self.video_writer.write(processed_frame)
                
                # 检查按键事件
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("收到退出按键，停止运行")
                    self.running = False
                    break
                elif key == ord('s'):
                    # 切换风格
                    self._switch_style()
                elif key == ord('+') or key == ord('='):
                    # 提高分辨率
                    self._increase_resolution()
                elif key == ord('-') or key == ord('_'):
                    # 降低分辨率
                    self._decrease_resolution()
                elif key == ord('r'):
                    # 开始/停止录制
                    self._toggle_recording(frame.shape)
        except KeyboardInterrupt:
            logger.info("收到键盘中断，退出程序")
            self.running = False
        except Exception as e:
            logger.error(f"程序运行出错: {e}")
            self.running = False
        finally:
            # 确保关闭视频写入器
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.shutdown()
        
        return True
    
    def _switch_style(self):
        """切换动漫风格
        
        使用键盘快捷键 's' 切换不同的预训练权重，实现不同动漫风格
        """
        try:
            # 切换到下一个权重
            self.current_weight_idx = (self.current_weight_idx + 1) % len(self.weight_list)
            next_weight_path = self.weight_list[self.current_weight_idx]
            
            logger.info(f"切换到风格 {self.current_weight_idx+1}/{len(self.weight_list)}，权重文件: {next_weight_path}")
            
            # 加载新权重
            self._load_animegan_model(next_weight_path)
        except Exception as e:
            logger.error(f"切换风格失败: {e}")
    
    def _increase_resolution(self):
        """提高输入分辨率
        
        使用键盘快捷键 '+' 提高分辨率，最高为512x512
        """
        if self.input_resolution < 512:
            self.input_resolution += 64
            logger.info(f"提高分辨率到 {self.input_resolution}x{self.input_resolution}")
        else:
            logger.info("已达到最高分辨率 512x512")
    
    def _decrease_resolution(self):
        """降低输入分辨率
        
        使用键盘快捷键 '-' 降低分辨率，最低为256x256
        """
        if self.input_resolution > 256:
            self.input_resolution -= 64
            logger.info(f"降低分辨率到 {self.input_resolution}x{self.input_resolution}")
        else:
            logger.info("已达到最低分辨率 256x256")
    
    def _toggle_recording(self, frame_shape):
        """开始/停止视频录制
        
        Args:
            frame_shape: 视频帧的形状，用于设置录制参数
        """
        try:
            if not self.is_recording:
                # 开始录制
                self.is_recording = True
                
                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                height, width = frame_shape[:2]
                output_path = f"anime_face_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                
                self.video_writer = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))
                logger.info(f"开始录制视频，保存路径: {output_path}")
            else:
                # 停止录制
                self.is_recording = False
                
                if self.video_writer is not None:
                    self.video_writer.release()
                    self.video_writer = None
                    logger.info("停止录制视频")
        except Exception as e:
            logger.error(f"视频录制操作失败: {e}")
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
    
    def shutdown(self):
        """关闭程序"""
        try:
            self.running = False
            
            # 关闭摄像头
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            
            # 关闭视频写入器
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            # 关闭所有窗口
            cv2.destroyAllWindows()
            
            logger.info("人脸实时动漫化程序已关闭")
        except Exception as e:
            logger.error(f"关闭程序失败: {e}")

# 命令行运行入口
if __name__ == "__main__":
    logger.info("=== 人脸实时动漫化程序启动 ===")
    
    # 使用本地权重文件
    weight_path = "../weights/face_paint_512_v2.pt"  # 默认使用 face_paint_512_v2.pt 权重
    
    try:
        # 创建实例
        face_to_anime = FaceToAnime(weight_path)
        
        # 运行程序
        face_to_anime.run()
    except Exception as e:
        logger.error(f"程序启动失败: {e}")
    
    logger.info("=== 人脸实时动漫化程序退出 ===")