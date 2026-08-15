from core import capture
from core import detector
from core import target_logic
from core import trajectory
from core import mouse_driver

import pyautogui
import keyboard
import time
import numpy as np


def main():
    # 捕获区域范围size*size
    size = 1440
    # 屏幕捕获帧率
    fps = 60
    # 推理输入尺寸；.engine 固定为 1440，请勿随意修改；.pt 可尝试调小到 640/800 提速
    img_size = 1440
    # 响应范围
    fov = 300
    # 获得屏幕大小
    screen_width, screen_height = pyautogui.size()
    # 截图区域内中心坐标
    center_x, center_y = size // 2, size // 2
    # 指定模型路径
    model_path = r"models/best_7.engine"
    # 识别类别
    _class = 0
    # 最低满足置信度
    min_conf = 0.7
    # 平滑系数
    smooth_factor = 0.8

    # ===== 性能优化开关 =====
    # 调试预览会显著增加端到端延迟，默认关闭
    SHOW_PREVIEW = False
    # 每 60 帧打印一次耗时，默认关闭
    PRINT_PERF = False
    # 鼠标移动死区（像素），避免准星在目标附近时频繁微调
    DEADZONE = 1.0
    # 推理时最多保留的检测框数量，降低 NMS 后处理开销
    MAX_DET = 20

    # 初始化相机实例
    camera = capture.init_camera(size, fps, screen_width, screen_height)
    # 初始化yolo模型实例
    model = detector.my_load_model(model_path)

    # 模型预热：避免第一帧推理因 TensorRT context 初始化而卡顿
    warmup_frame = np.zeros((size, size, 3), dtype=np.uint8)
    detector.predict_target(
        model,
        warmup_frame,
        img_size=img_size,
        conf=min_conf,
        classes=[_class],
        max_det=MAX_DET,
    )

    # --- 性能统计变量 ---
    frame_count = 0
    total_capture_time = 0.0
    total_infer_time = 0.0
    total_logic_time = 0.0
    total_mouse_time = 0.0
    total_loop_time = 0.0

    if PRINT_PERF:
        print("开始运行并统计性能，每 60 帧输出一次平均耗时...")

    while True:
        # 监听是否按下退出
        if keyboard.is_pressed('f12'):
            print('按下退出键，退出')
            break

        loop_start = time.perf_counter() if PRINT_PERF else None

        # 1. 测量截图耗时
        t0 = time.perf_counter() if PRINT_PERF else None
        frame = capture.my_get_latest_frame(camera)
        t1 = time.perf_counter() if PRINT_PERF else None

        if frame is None:
            continue

        # 2. 测量推理耗时
        boxes = detector.predict_target(
            model,
            frame,
            img_size=img_size,
            conf=min_conf,
            classes=[_class],
            max_det=MAX_DET,
        )
        t2 = time.perf_counter() if PRINT_PERF else None

        # 3. 测量逻辑筛选耗时
        target_x, target_y = target_logic.find_best_target(
            boxes, center_x, center_y, fov, _class, min_conf
        )
        t3 = time.perf_counter() if PRINT_PERF else None

        # 4. 测量鼠标移动耗时
        if target_x is not None:
            move_x, move_y = trajectory.calculate_movement(
                target_x, target_y, center_x, center_y, smooth_factor
            )
            # 死区过滤，避免目标已经接近准星时仍发送无意义微移
            if abs(move_x) > DEADZONE or abs(move_y) > DEADZONE:
                mouse_driver.move_mouse(move_x, move_y)

        # 5. 演示视频渲染模块（默认关闭，开启会明显增加延迟）
        if SHOW_PREVIEW:
            import cv2

            if target_x is not None and boxes is not None and len(boxes) > 0:
                # 将 GPU 上的坐标数据转到 CPU 并提取
                xyxy_array = boxes.xyxy.cpu().numpy()
                for box in xyxy_array:
                    x1, y1, x2, y2 = map(int, box[:4])
                    # 画出所有敌人的绿色边框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 用一根红线将准星和锁定的目标连接起来
                cv2.line(frame, (center_x, center_y), (int(target_x), int(target_y)), (0, 0, 255), 2)
                # 在锁定目标身上画一个醒目的红色准心
                cv2.circle(frame, (int(target_x), int(target_y)), 5, (0, 0, 255), -1)

            show_frame = cv2.resize(frame, (600, 450)) if size > 800 else frame
            cv2.imshow("Aimbot Demo", show_frame)
            # 这里的 pollKey() 防止窗口卡死或闪烁
            cv2.pollKey()

        t4 = time.perf_counter() if PRINT_PERF else None

        # 累加各部分耗时 (乘以 1000 将秒转换为毫秒)
        if PRINT_PERF:
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
                print(f"  └─ 鼠标+渲染: {avg_mouse:.2f} ms\n")

                # 打印完后清零，重新开始下一轮 60 帧的统计
                total_capture_time = 0.0
                total_infer_time = 0.0
                total_logic_time = 0.0
                total_mouse_time = 0.0
                total_loop_time = 0.0

    return


if __name__ == "__main__":
    main()
