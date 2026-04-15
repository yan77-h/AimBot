from core import capture
from core import detector
from core import target_logic
from core import trajectory
from core import mouse_driver
import pyautogui
import keyboard

def main():
    # 判断是否按下退出键，按下即退出程序
    is_paused = False
    # 捕获区域范围size*size
    size = 800
    # 屏幕捕获帧率
    fps = 60
    # 响应范围
    fov = 400
    # 获得屏幕大小
    screen_width, screen_height = pyautogui.size()
    # 截图区域内中心坐标
    center_x, center_y = size//2, size//2
    # 指定模型路径
    model_path = r"models/best_7.pt"
    # 识别类别
    _class = 0
    # 最低满足置信度
    min_conf = 0
    # 平滑系数
    smooth_factor = 1

    # 初始化相机实例
    camera = capture.init_camera(size,fps,screen_width,screen_height)
    # 初始化yolo模型实例
    model = detector.my_load_model(model_path)

    while True:
        # 监听是否按下退出
        if keyboard.is_pressed('f12'):
            print('按下退出键，退出')
            break

        # 获得最新一帧图像
        frame = capture.my_get_latest_frame(camera)

        # 进行预测，得到坐标信息
        boxes = detector.predict_target(model, frame)

        # 解析得到坐标（坐标系为捕获区域）
        target_x, target_y = target_logic.find_best_target(boxes, center_x, center_y, fov, _class, min_conf)

        if target_x is None:
            continue

        # 得到相对位移，经过平滑处理
        move_x, move_y = trajectory.calculate_movement(target_x, target_y, center_x, center_y, smooth_factor)

        # 移动鼠标
        mouse_driver.move_mouse(move_x, move_y)


    return

if __name__ == "__main__":
    main()