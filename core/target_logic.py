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

    # 2. 将数据从 GPU 提取到 CPU 并转为 NumPy 数组
    xywh_array = boxes.xywh.cpu().numpy()
    cls_array = boxes.cls.cpu().numpy()
    conf_array = boxes.conf.cpu().numpy()

    # 3. 用向量化方式一次性完成类别和置信度过滤
    mask = (cls_array == target_class) & (conf_array >= min_conf)
    if not np.any(mask):
        return None, None

    candidates = xywh_array[mask]

    # 4. 计算所有候选目标到准星中心的距离平方（避免开根号）
    distances_sq = ((candidates[:, 0] - center_x) ** 2 +
                    (candidates[:, 1] - center_y) ** 2)

    # 5. 过滤超出 FOV 的目标
    max_fov_sq = max_fov * max_fov
    in_fov = distances_sq <= max_fov_sq
    if not np.any(in_fov):
        return None, None

    # 6. 取最近的目标
    best_idx = int(np.argmin(distances_sq[in_fov]))
    best_x, best_y = candidates[in_fov][best_idx][:2]

    return float(best_x), float(best_y)
