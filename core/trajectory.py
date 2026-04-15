# 弹道与平滑模块

"""
计算目标与准星中心的差值，并作处理
接收 目标坐标 准星坐标 系数
返回处理后鼠标要移动的像素值
"""
def calculate_movement(target_x, target_y, center_x, center_y, factor):
    original_move_x = target_x - center_x
    original_move_y = target_y - center_y

    move_x = original_move_x * factor
    move_y = original_move_y * factor

    return move_x, move_y

