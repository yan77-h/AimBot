from ultralytics import YOLO

# 目标检测模块

"""
加载yolo模型
接收模型路径
返回模型实例
"""
def my_load_model(model_path):
    model = YOLO(model_path)

    return model

"""
将图像送入模型推理，得到坐标数据
接收 模型实例 图像 推理尺寸
返回boxes
"""
def predict_target(model, frame, img_size=1440):
    results = model.predict(frame, imgsz=img_size, rect=True, verbose=False)
    boxes = results[0].boxes

    return boxes