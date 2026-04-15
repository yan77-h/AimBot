# 🎯 AimBot - 基于 YOLO11n 的 FPS 游戏实时自瞄助手

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![YOLO](https://img.shields.io/badge/YOLO-11n-orange)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Platform](https://img.shields.io/badge/platform-Windows%2010%2F11-lightgrey)]()

## 📖 项目简介

这是一个基于 **YOLOv11** 目标检测模型的 FPS 游戏智能辅助工具。通过实时捕获屏幕中心区域画面，识别敌方目标，并自动将鼠标平滑移动至目标位置，实现辅助瞄准。

> ⚠️ **重要声明**：本项目**仅供计算机视觉技术学习与交流**，严禁在正式游戏对战中使用，使用本项目造成的一切账号封禁或法律后果由使用者自行承担。

### ✨ 核心特性
- **实时屏幕捕获**：基于 `DXCam` 实现高帧率、低延迟的屏幕区域捕获。
- **YOLO11n 推理**：利用 YOLO 模型进行高效目标检测。
- **智能目标筛选**：支持按类别、置信度、视野范围过滤，并自动锁定最近目标。
- **平滑鼠标移动**：相对移动指令配合平滑系数，使准星移动更自然。
- **模块化设计**：画面捕获、模型推理、逻辑判断、硬件控制完全解耦，便于二次开发。

## 🛠️ 技术栈

- **Python** 3.9+
- **YOLO11n** ([Ultralytics](https://github.com/ultralytics/ultralytics))
- **DXCam** (高性能屏幕捕获)
- **PyDirectInput** (模拟鼠标输入)
- **OpenCV & NumPy** (图像与矩阵运算)
- **Keyboard** (全局热键监听)

## 📂 项目结构

```text
.
├── main.py                # 主程序入口，负责参数配置与主循环
├── core/
│   ├── capture.py         # 屏幕捕获模块 (DXCam 封装)
│   ├── detector.py        # YOLO 模型加载与推理模块
│   ├── target_logic.py    # 目标筛选逻辑 (FOV、置信度、最近距离)
│   ├── trajectory.py      # 坐标转换与平滑计算
│   └── mouse_driver.py    # 鼠标相对移动控制
├── models/
│   └── best.pt            # 训练好的 YOLOv11 权重文件
└── environment.yml        # Conda 环境配置文件
```

## 🚀 快速开始

### 环境要求
- Windows 10 / 11
- 支持 CUDA 的 NVIDIA 显卡（项目使用的显卡 NVIDIA GeForce RTX 4060 Laptop GPU, 8188MiB）
- Anaconda 或 Miniconda 已安装

### 安装步骤

**注意！以下环境对应于开发者设备的硬件环境，不一定适用！若不适用，请自行配置适合于本地硬件的环境，一般注意 cuda 和 pytorch 的版本即可**

1. **克隆仓库**
   
   ```bash
   git clone https://github.com/yan77-h/AimBot.git
   cd AimBot
   ```
   
2. **使用 Conda 创建并激活环境**

   项目根目录已提供 `environment.yml` 文件，可直接复现完全相同的开发环境：

   ```bash
   conda env create -f environment.yml
   conda activate yolo
   ```

3. **放置模型权重(可选)**
   将你训练好的 YOLO 模型文件 (`best.pt`) 放入 `models/` 文件夹。

### 运行程序

1. 打开你要测试的游戏，并设置为**窗口化**或**无边框窗口模式**。
2. 在终端运行主程序：
   ```bash
   python main.py
   ```
3. **默认操作**：
   
   - 程序自动对屏幕中央 `800x800` 区域进行推理。
   - 当检测到 `类别 0` 且距离准星最近的敌人时，鼠标会自动向其移动。
   - **按下 `F12` 键**即可安全退出程序。

## ⚙️ 配置说明 (main.py 参数)

你可以在 `main.py` 中直接修改以下变量来调整自瞄行为：

| 变量名 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `size` | `800` | 屏幕捕获区域的边长 (像素) |
| `fps` | `60` | 目标捕获帧率 |
| `fov` | `400` | 最大响应范围 (距离屏幕中心的半径) |
| `_class` | `0` | 要锁定的目标类别 ID (0为head，1为body) |
| `min_conf` | `0` | 最低置信度阈值 (0~1) |
| `smooth_factor` | `1` | 平滑系数 (越小移动越慢/越平滑) |

## 🤝 贡献与改进计划

欢迎对代码逻辑提出优化建议！目前待办事项包括：
- [ ] 增加图形化设置界面 (GUI) 方便调整参数
- [ ] 优化 CPU 占用与推理延迟
- [ ] 改用效果更好的模型
- [ ] 改用 TensorRT 格式模型减少延迟

## 📄 许可证

本项目基于 MIT 协议开源。请查阅 `LICENSE` 文件获取详细信息。

## 📧 联系方式

如有问题或建议，欢迎提 Issue 或通过以下方式联系：
- GitHub: [@yan77-h](https://github.com/yan77-h)
- Email: yan_@tju.edu.cn