# 实时人脸动漫化项目

## 项目概述

本项目是一个基于AnimeGANv2的实时人脸动漫化程序，能够通过摄像头实时捕捉人脸，并将其转换为精美的动漫风格。该项目使用了先进的深度学习技术，能够在保持实时性的同时，生成高质量的动漫化效果。项目经过优化，支持多种动漫风格切换、实时视频录制和自适应分辨率调整。

## 功能特点

1. **实时处理**：通过摄像头实时捕捉人脸并进行动漫化处理
2. **高质量效果**：使用AnimeGANv2模型，生成高质量的动漫风格图像
3. **支持GPU加速**：自动检测并使用GPU（如果可用），提高处理速度
4. **高效人脸检测**：使用MTCNN人脸检测器，提高检测精度和速度
5. **多风格切换**：支持4种不同动漫风格，可通过键盘快捷键实时切换
6. **自适应分辨率**：支持256x256到512x512分辨率调整，平衡效果和速度
7. **优化的1280x720支持**：针对1280x720摄像头分辨率进行了优化，确保最佳效果
8. **视频录制功能**：支持实时录制动漫化视频，保存为MP4格式
9. **自然融合效果**：使用泊松融合算法，实现人脸与背景的自然融合
10. **简单易用**：只需运行一个命令即可启动程序
11. **跨平台**：支持Windows、Linux和macOS等主流操作系统

## 技术栈

- **Python**：项目开发语言
- **OpenCV**：用于摄像头操作和图像处理
- **PyTorch**：用于深度学习模型的加载和推理
- **AnimeGANv2**：用于人脸动漫化的预训练模型
- **MTCNN**：用于高精度人脸检测
- **TensorFlow**：MTCNN依赖的深度学习框架
- **NumPy**：用于数值计算和数组操作

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/anime-face.git
cd anime-face
```

### 2. 创建虚拟环境

```bash
# 使用conda创建虚拟环境
conda create -n anime-face python=3.11
conda activate anime-face

# 或使用venv创建虚拟环境
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 下载预训练权重

将预训练权重文件下载到`weights`目录下，支持以下权重：
- `celeba_distill.pt`：适合生成卡通风格
- `face_paint_512_v1.pt`：适合生成手绘风格
- `face_paint_512_v2.pt`：适合生成更精细的手绘风格
- `paprika.pt`：适合生成动漫风格

## 使用方法

### 运行程序

```bash
python scripts/main.py
```

程序将自动打开摄像头，开始实时人脸动漫化处理。

### 摄像头分辨率

程序默认使用1280x720分辨率，这是作者的硬件摄像头分辨率，已经过专门优化，能够提供最佳的实时处理效果。

如果您的摄像头不支持1280x720分辨率，程序会自动调整为摄像头支持的最接近分辨率。

您可以在代码中修改默认分辨率：

```python
# 在init_camera方法中修改默认参数
def init_camera(self, camera_id: int = 0, width: int = 1280, height: int = 720) -> bool:
    # ...
```

### 键盘快捷键

程序运行时，支持以下键盘快捷键：

| 快捷键 | 功能描述 |
|--------|----------|
| `q`    | 退出程序 |
| `s`    | 切换动漫风格（4种风格循环切换） |
| `+`    | 提高输入分辨率（最高512x512） |
| `-`    | 降低输入分辨率（最低256x256） |
| `r`    | 开始/停止视频录制 |

### 状态显示

程序运行时，界面顶部会显示当前状态信息：

```
风格: X/Y | 分辨率: WxH | 录制: 开启/关闭
```

其中：
- `X/Y`：当前风格编号/总风格数
- `WxH`：当前输入分辨率
- `开启/关闭`：视频录制状态

### 视频保存

录制的视频将保存在当前目录下，文件名格式为：

```
anime_face_YYYYMMDD_HHMMSS.mp4
```

其中`YYYYMMDD_HHMMSS`是录制开始的时间戳。

## 项目结构

```
anime-face/
├── scripts/              # 主要代码目录
│   ├── main.py          # 主程序入口
│   └── analyze_weights.py  # 权重分析脚本
├── weights/             # 预训练权重目录
│   ├── celeba_distill.pt
│   ├── face_paint_512_v1.pt
│   ├── face_paint_512_v2.pt
│   └── paprika.pt
├── requirements.txt     # 依赖列表
└── README.md           # 项目说明文档
```

## 核心代码说明

### 1. 模型定义

项目使用了完整的AnimeGANv2模型结构，包括ConvNormLReLU和InvertedResBlock类：

```python
class AnimeGANV2(nn.Module):
    def __init__(self, ):
        super().__init__()
        # 定义模型结构
        self.block_a = nn.Sequential(...)
        self.block_b = nn.Sequential(...)
        self.block_c = nn.Sequential(...)
        self.block_d = nn.Sequential(...)
        self.block_e = nn.Sequential(...)
        self.out_layer = nn.Sequential(...)
    
    def forward(self, input, align_corners=True):
        # 模型前向传播
        out = self.block_a(input)
        out = self.block_b(out)
        out = self.block_c(out)
        # 上采样和后续处理
        # ...
        return out
```

### 2. MTCNN人脸检测

使用MTCNN进行高精度人脸检测：

```python
def _get_face_bbox(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """使用MTCNN检测人脸边界框"""
    try:
        # MTCNN检测人脸
        boxes = self.mtcnn.detect_faces(frame)
        if not boxes:
            return None
        # 获取第一个检测到的人脸边界框
        box = boxes[0]['box']
        x, y, w, h = box
        return (x, y, w, h)
    except Exception as e:
        logger.error(f"检测人脸失败: {e}")
        return None
```

### 3. 自适应分辨率处理

支持动态调整输入分辨率，平衡效果和速度：

```python
def _animegan_process(self, face_region: np.ndarray) -> np.ndarray:
    """使用AnimeGANv2处理人脸区域"""
    # 使用可调整的输入分辨率
    face_resized = cv2.resize(face_region, (self.input_resolution, self.input_resolution))
    # 预处理和推理
    # ...
    return anime_face
```

### 4. 泊松融合算法

实现自然的人脸融合效果：

```python
def _blend_anime_face(self, frame: np.ndarray, anime_face: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
    """使用泊松融合将动漫化后的人脸融合回原图"""
    # 创建掩码和ROI区域
    mask = np.ones((h_crop, w_crop), dtype=np.uint8) * 255
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    roi = frame[y:y_end, x:x_end]
    
    try:
        # 使用泊松融合实现自然融合
        center = (w_crop // 2, h_crop // 2)
        blended_face = cv2.seamlessClone(anime_face_crop, roi, mask, center, cv2.NORMAL_CLONE)
        frame[y:y_end, x:x_end] = blended_face
    except Exception as e:
        # 备用方案：简单加权融合
        # ...
    return frame
```

### 5. 多风格切换

支持通过键盘快捷键切换不同动漫风格：

```python
def _switch_style(self):
    """切换动漫风格"""
    # 切换到下一个权重
    self.current_weight_idx = (self.current_weight_idx + 1) % len(self.weight_list)
    next_weight_path = self.weight_list[self.current_weight_idx]
    # 加载新权重
    self._load_animegan_model(next_weight_path)
```

### 6. 视频录制功能

支持实时录制动漫化视频：

```python
def _toggle_recording(self, frame_shape):
    """开始/停止视频录制"""
    if not self.is_recording:
        # 开始录制
        self.is_recording = True
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = f"anime_face_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.video_writer = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))
    else:
        # 停止录制
        self.is_recording = False
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
```

## 优化建议

### 已实现的优化

✅ **使用更高效的人脸检测器**：已实现MTCNN人脸检测器，提高了检测精度和速度

✅ **优化模型推理速度**：已实现自适应分辨率调整，支持256x256到512x512分辨率切换

✅ **添加风格切换功能**：已实现4种风格切换，支持键盘快捷键实时切换

✅ **添加视频录制功能**：已实现视频录制功能，支持MP4格式保存

✅ **优化人脸融合效果**：已实现泊松融合算法，实现自然的人脸融合

### 未来可考虑的优化

1. **使用ONNX Runtime或TensorRT加速推理**：
   - 将PyTorch模型转换为ONNX格式
   - 使用ONNX Runtime或TensorRT进行推理加速
   - 可进一步提高处理速度，支持更高分辨率

2. **添加更多动漫风格**：
   - 收集和训练更多风格的预训练权重
   - 支持自定义风格添加

3. **实现多人脸同时处理**：
   - 支持同时检测和处理多个人脸
   - 为每个人脸应用独立的动漫化处理

4. **添加人脸关键点动画**：
   - 检测人脸关键点
   - 根据关键点位置调整动漫化效果
   - 实现更生动的表情动画

5. **添加背景虚化功能**：
   - 实现背景虚化效果
   - 突出动漫化人脸主体

6. **支持图片和视频输入**：
   - 支持从图片文件输入
   - 支持从视频文件输入
   - 批量处理功能

7. **添加GUI界面**：
   - 实现图形用户界面
   - 提供可视化参数调整
   - 支持拖拽式操作

8. **移动端部署**：
   - 考虑将模型部署到移动端
   - 实现边缘设备实时处理

## 总结

本项目成功实现了一个功能丰富的实时人脸动漫化系统，基于AnimeGANv2模型，能够通过摄像头实时捕捉人脸并转换为精美的动漫风格。项目具有以下特点：

1. **高质量的动漫化效果**：使用AnimeGANv2模型，生成细腻、生动的动漫风格图像
2. **高效的人脸检测**：采用MTCNN人脸检测器，确保在各种场景下都能准确检测人脸
3. **丰富的交互功能**：支持风格切换、分辨率调整和视频录制等多种功能
4. **自然的融合效果**：使用泊松融合算法，实现人脸与背景的无缝融合
5. **良好的用户体验**：简洁的界面设计和直观的键盘操作

通过本项目的开发，我们深入学习了：

- AnimeGANv2模型的结构和工作原理
- MTCNN人脸检测算法的应用
- 实时图像处理和视频流处理技术
- 泊松融合等高级图像处理算法
- 多线程编程和资源管理

项目已经过充分测试，能够稳定运行，并且具有良好的扩展性。未来可以进一步优化性能，添加更多功能，使其在更多场景下得到应用。

## 应用场景

1. **视频会议和直播**：为视频会议或直播添加动漫化效果，保护隐私的同时增加趣味性
2. **社交媒体内容创作**：生成独特的动漫风格头像和短视频
3. **游戏和虚拟形象**：为游戏角色或虚拟形象提供实时动漫化效果
4. **教育和娱乐**：用于教育演示或娱乐应用，增加互动性和趣味性
5. **创意设计**：为设计师提供灵感，生成独特的动漫风格素材

本项目展示了深度学习技术在实时图像处理领域的强大应用潜力，为相关研究和开发提供了有价值的参考。

## 参考资料

1. [AnimeGANv2 GitHub仓库](https://github.com/bryandlee/animegan2-pytorch)
2. [PyTorch官方文档](https://pytorch.org/docs/stable/index.html)
3. [OpenCV官方文档](https://docs.opencv.org/4.x/)
4. [MediaPipe官方文档](https://google.github.io/mediapipe/)

## 许可证

本项目采用MIT许可证，详见LICENSE文件。