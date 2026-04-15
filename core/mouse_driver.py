import pydirectinput

# 硬件与输入模块

"""
初始化
"""
# def init_mouse():
#     return

"""
移动鼠标
接收鼠标位移坐标
无返回
"""
def move_mouse(move_x, move_y):
    move_x, move_y = int(move_x), int(move_y)
    pydirectinput.move(move_x, move_y, relative=True, _pause=False)

    return

"""
检测是否触发自瞄，监听键盘
接收按键代码
返回bool
"""
# def is_aim_key_pressed(key_code):
#
#     return
