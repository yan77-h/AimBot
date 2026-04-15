import dxcam

# 画面捕获模块

"""
初始化截图实例，设定截图区域
接收值为截图区域，帧率，屏幕宽高
返回值为实例化的相机对象
"""
def init_camera(capture_size, capture_fps, screen_width, screen_height):
    # 计算得到大小为capture_size的屏幕中间区域的方位
    left, top = (screen_width - capture_size) // 2, (screen_height - capture_size) // 2
    right, bottom = left + capture_size, top + capture_size
    region = (left, top, right, bottom)

    # 启动实例
    camera = dxcam.create(output_color = 'BGR')
    camera.start(target_fps=capture_fps, video_mode=True, region=region)

    return camera

"""
从相机获取最新的一帧画面
接收实例化的相机对象
返回图像
"""
def my_get_latest_frame(camera):
    return camera.get_latest_frame()


