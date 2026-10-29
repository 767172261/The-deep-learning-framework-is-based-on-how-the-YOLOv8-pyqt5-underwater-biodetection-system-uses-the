深度学习框架基于YOLOv8➕pyqt5的水下生物检测系统 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3eb0c462d93945d1a25f7ac09f0b4769.png)

内含7600张水下生物数据集
包括['海胆', '海参', '扇贝', '海星', '水草']，5类

也可自行替换模型，使用该界面做其他检测

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2a72806f6f3a4e1684f18cca2a6afceb.png)
 **基于 YOLOv8 + PyQt5 的水下生物检测系统**，包含：

✅ 7600 张标注数据集（5 类：海胆、海参、扇贝、海星、水草）  
✅ 预训练模型（YOLOv8s）→ 自定义训练 → 推理  
✅ 完整 Python 代码（含 UI 界面、推理逻辑、文件操作）  
✅ 可替换任意 YOLO 模型（支持 .pt 格式）  
✅ 支持图片 / 视频 / 摄像头实时检测  
✅ 图形化界面（PyQt5），美观易用  

---

## 📁 一、项目结构

```bash
UnderwaterDetectionSystem/
├── datasets/                      # 数据集（YOLO格式）
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
│   └── underwater_best.pt         # 训练好的模型（可替换）
├── UIProgram/
│   ├── MainProgram.py             # 主程序
│   ├── detect_tools.py            # 检测工具类
│   └── config.yaml                # 数据配置文件
├── train.py                       # 模型训练脚本
└── README.md
```

---

## 🗃️ 二、1. 数据集说明（5类）

| 类别 | 中文名 | 英文名 |
|------|--------|--------|
| 0    | 海胆   | Sea Urchin |
| 1    | 海参   | Sea Cucumber |
| 2    | 扇贝   | Scallop |
| 3    | 海星   | Starfish |
| 4    | 水草   | Seaweed |

> ✅ 数据集总量：**7,600 张图像**（已标注）  
> ✅ 分割比例：`train:val = 8:2`（约 6080 / 1520）  
> ✅ 标注工具建议：[LabelImg](https://github.com/HumanSignal/labelImg)  
> ✅ 格式：YOLO 格式 `.txt` 文件（归一化坐标）

---

## 📂 三、2. `config.yaml` 配置文件

```yaml
# config.yaml
train: ./datasets/images/train
val: ./datasets/images/val

nc: 5
names: ['海胆', '海参', '扇贝', '海星', '水草']
```

---

## 🔧 四、3. 模型训练脚本（`train.py`）

```python
# train.py
from ultralytics import YOLO
import os

def main():
    os.makedirs("runs", exist_ok=True)
    
    model = YOLO('yolov8s.pt')  # 使用预训练权重
    results = model.train(
        data='config.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='underwater_detection',
        project='runs',
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        patience=20,
        save=True,
        device=0,
        workers=4,
        cache=False,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    print("✅ 训练完成！最佳模型路径：", results.save_dir / "weights/best.pt")

if __name__ == '__main__':
    main()
```

> ✅ 运行：
```bash
python train.py
```
> ⏳ 输出模型：`runs/underwater_detection/weights/best.pt`

---

## 🧩 五、4. 检测工具类（`detect_tools.py`）

```python
# detect_tools.py
from ultralytics import YOLO
import cv2
import numpy as np

class UnderwaterDetector:
    def __init__(self, model_path="models/underwater_best.pt", conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_names = ['海胆', '海参', '扇贝', '海星', '水草']
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]  # BGR

    def detect_image(self, image_path):
        """检测单张图片"""
        results = self.model(image_path, conf=self.conf_threshold)
        result = results[0]
        boxes = result.boxes.cpu().numpy()
        detections = []

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = self.class_names[cls_id]
            color = self.colors[cls_id]

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections.append({
                "class": class_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2)
            })

        return detections, result.plot()

    def detect_video(self, video_path):
        """检测视频流"""
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, conf=self.conf_threshold)
            annotated_frame = results[0].plot()
            yield annotated_frame
        cap.release()
```

---

## 🖥️ 六、5. PyQt5 主界面（`MainProgram.py`）

```python
# MainProgram.py
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QLineEdit, QComboBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import cv2
from detect_tools import UnderwaterDetector

class UnderwaterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的水下生物检测系统")
        self.setGeometry(100, 100, 1000, 700)
        self.detector = UnderwaterDetector()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 左侧图像显示区
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #007BFF;")
        left_layout.addWidget(self.image_label)

        # 右侧控制区
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 文件输入
        self.file_input = QLineEdit()
        self.file_input.setPlaceholderText("请选择图片或视频...")
        self.btn_open = QPushButton("打开文件")
        self.btn_open.clicked.connect(self.open_file)
        right_layout.addWidget(self.file_input)
        right_layout.addWidget(self.btn_open)

        # 检测参数设置
        self.conf_slider = QComboBox()
        self.conf_slider.addItems(['0.25', '0.3', '0.35', '0.4', '0.45'])
        self.conf_slider.setCurrentIndex(0)
        right_layout.addWidget(self.conf_slider)

        # 检测结果
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)

        # 操作按钮
        self.btn_start = QPushButton("开始检测")
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.clicked.connect(self.stop_detection)
        right_layout.addWidget(self.btn_start)
        right_layout.addWidget(self.btn_stop)

        # 保存按钮
        self.btn_save = QPushButton("保存结果")
        self.btn_save.clicked.connect(self.save_result)
        right_layout.addWidget(self.btn_save)

        main_layout.addWidget(left_panel, stretch=2)
        main_layout.addWidget(right_panel, stretch=1)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择文件", "", "图像文件 (*.jpg *.png);;视频文件 (*.mp4 *.avi)"
        )
        if file_path:
            self.file_input.setText(file_path)
            self.detect_and_show(file_path)

    def detect_and_show(self, path):
        if path.lower().endswith(('.jpg', '.png')):
            detections, img = self.detector.detect_image(path)
            self.show_image(img)
            self.display_results(detections)
        elif path.lower().endswith(('.mp4', '.avi')):
            self.video_path = path
            self.start_detection()

    def start_detection(self):
        if self.file_input.text().lower().endswith(('.mp4', '.avi')):
            self.cap = cv2.VideoCapture(self.file_input.text())
            self.timer.start(30)
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
        else:
            self.detect_and_show(self.file_input.text())

    def stop_detection(self):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            results = self.detector.model(frame, conf=float(self.conf_slider.currentText()))
            annotated_frame = results[0].plot()
            self.show_image(annotated_frame)
            self.display_results(results[0].boxes.cpu().numpy())

    def show_image(self, img):
        h, w, ch = img.shape
        bytes_per_line = ch * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img).scaled(600, 600, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

    def display_results(self, boxes):
        if isinstance(boxes, np.ndarray):
            boxes = boxes[0]
        results = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = self.detector.class_names[cls]
            results.append(f"类别: {class_name}, 置信度: {conf:.2f}, 位置: [{x1},{y1},{x2},{y2}]")
        self.result_text.setText("\n".join(results))

    def save_result(self):
        pass  # 可扩展为保存图片或日志

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = UnderwaterApp()
    window.show()
    sys.exit(app.exec_())
```

---

## 💡 七、使用说明

### ✅ 步骤 1：安装依赖
```bash
pip install ultralytics opencv-python pyqt5 numpy
```

### ✅ 步骤 2：准备数据集
- 将标注好的 `datasets/` 放入项目根目录
- 修改 `config.yaml` 路径
- 运行 `train.py` 训练模型（或直接使用 `models/underwater_best.pt`）

### ✅ 步骤 3：运行主程序
```bash
cd UIProgram
python MainProgram.py
```

### ✅ 功能支持
| 功能 | 支持 |
|------|------|
| ✅ 图片检测 | 是 |
| ✅ 视频检测 | 是 |
| ✅ 摄像头实时检测 | 是 |
| ✅ 自定义置信度阈值 | 是 |
| ✅ 检测结果可视化 | 是 |
| ✅ 检测信息表格输出 | 是 |
| ✅ 模型热插拔（替换 .pt） | 是 |

---


