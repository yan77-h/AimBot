from core import capture
from core import detector
from core import target_logic
from core import trajectory
from core import mouse_driver
import pyautogui
import keyboard
import time

def main():
    # 判断是否按下退出键，按下即退出程序
    is_paused = False
    # 捕获区域范围size*size
    size = 1440
    # 屏幕捕获帧率
    fps = 60
    # 响应范围
    fov = 300
    # 获得屏幕大小
    screen_width, screen_height = pyautogui.size()
    # 截图区域内中心坐标
    center_x, center_y = size//2, size//2
    # 指定模型路径
    model_path = r"models/best_7.engine"
    # 识别类别
    _class = 0
    # 最低满足置信度
    min_conf = 0.7
    # 平滑系数
    smooth_factor = 0.8

    # 初始化相机实例
    camera = capture.init_camera(size,fps,screen_width,screen_height)
    # 初始化yolo模型实例
    model = detector.my_load_model(model_path)

    # --- 性能测试专用变量 ---
    frame_count = 0
    total_capture_time = 0
    total_infer_time = 0
    total_logic_time = 0
    total_mouse_time = 0
    total_loop_time = 0
    print("开始运行并统计性能，每 60 帧输出一次平均耗时...")

    while True:
        # 监听是否按下退出
        if keyboard.is_pressed('f12'):
            print('按下退出键，退出')
            break

        loop_start = time.perf_counter()  # 记录本轮循环总开始时间

        # 1. 测量截图耗时
        t0 = time.perf_counter()
        frame = capture.my_get_latest_frame(camera)
        t1 = time.perf_counter()

        # 2. 测量推理耗时
        boxes = detector.predict_target(model, frame)
        t2 = time.perf_counter()

        # 3. 测量逻辑筛选耗时
        target_x, target_y = target_logic.find_best_target(boxes, center_x, center_y, fov, _class, min_conf)
        t3 = time.perf_counter()

        # 4. 测量鼠标移动耗时
        if target_x is not None:
            move_x, move_y = trajectory.calculate_movement(target_x, target_y, center_x, center_y, smooth_factor)
            mouse_driver.move_mouse(move_x, move_y)
        t4 = time.perf_counter()

        # 累加各部分耗时 (乘以 1000 将秒转换为毫秒)
        total_capture_time += (t1 - t0) * 1000
        total_infer_time += (t2 - t1) * 1000
        total_logic_time += (t3 - t2) * 1000
        total_mouse_time += (t4 - t3) * 1000
        total_loop_time += (t4 - loop_start) * 1000
        frame_count += 1

        # 每隔 60 帧（大约1秒多），计算并打印一次平均值
        if frame_count % 60 == 0:
            avg_capture = total_capture_time / 60
            avg_infer = total_infer_time / 60
            avg_logic = total_logic_time / 60
            avg_mouse = total_mouse_time / 60
            avg_loop = total_loop_time / 60

            print(f"--- 过去 60 帧平均耗时 ---")
            print(f"总循环耗时: {avg_loop:.2f} ms (自瞄刷新率: {1000 / avg_loop:.1f} FPS)")
            print(f"  ├─ 截图: {avg_capture:.2f} ms")
            print(f"  ├─ 推理: {avg_infer:.2f} ms")
            print(f"  ├─ 逻辑: {avg_logic:.2f} ms")
            print(f"  └─ 鼠标: {avg_mouse:.2f} ms\n")

            # 打印完后清零，重新开始下一轮 60 帧的统计
            total_capture_time = 0
            total_infer_time = 0
            total_logic_time = 0
            total_mouse_time = 0
            total_loop_time = 0

    return

if __name__ == "__main__":
    main()