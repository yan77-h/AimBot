import numpy as np

# 目标筛选逻辑模块

"""
遍历boxes，过滤，计算距离，找出最近的目标
接收 boxes 截图区域中心点 范围 目标类别 置信度要求
返回最佳目标坐标
"""
def find_best_target(boxes, center_x, center_y, max_fov, target_class, min_conf):
    # 1. 检查是否有检测到任何目标
    if boxes is None or len(boxes) == 0:
        return None, None

    # 2. 将数据从 GPU 提取到 CPU 并转为 NumPy 数组，大幅提升后续循环提取速度
    xywh_array = boxes.xywh.cpu().numpy()
    cls_array = boxes.cls.cpu().numpy()
    conf_array = boxes.conf.cpu().numpy()

    best_target = None
    min_distance = float('inf')

    # 3. 遍历所有检测到的目标
    for i in range(len(boxes)):
        # 提取当前目标的详细数据
        box_class = int(cls_array[i])
        box_conf = conf_array[i]
        box_x, box_y, box_w, box_h = xywh_array[i]

        # 过滤 1：类别不符的跳过
        if box_class != target_class:
            continue

        # 过滤 2：置信度太低的跳过
        if box_conf < min_conf:
            continue

        # 4. 计算到准星中心（屏幕中心）的距离
        distance = np.sqrt((box_x - center_x) ** 2 + (box_y - center_y) ** 2)

        # 过滤 3：超出 FOV 圈的跳过
        if distance > max_fov:
            continue

        # 5. 更新最近目标
        if distance < min_distance:
            min_distance = distance
            best_target = (box_x, box_y)

    # 6. 返回结果
    if best_target:
        return best_target[0], best_target[1]
    else:
        return None, None
