import dxcam
import cv2
import pyautogui
from ultralytics import YOLO

model_name = r'E:\deeplearning\ultralytics-8.3.163\runs\detect\train4\weights\best.pt'
model = YOLO(model_name)

screen_width,screen_height = pyautogui.size()

center_x,center_y = screen_width/2,screen_height/2

left, top = (2560-800)//2, (1600-800)//2
right, bottom = left + 800, top + 800

region = (left, top, right, bottom)

camera = dxcam.create(output_color = 'BGR')

camera.start(target_fps=60, video_mode=True, region=region)

target_head = 0
target_body = 1

while True:
    frame = camera.get_latest_frame()
    if frame is not None:
        # cv2.imshow('frame', frame)

        #yolo检测代码
        results = model.predict(frame,imgsz=1440,rect=True)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            box_best = boxes[0]
            for box in boxes: #找到距离准心曼哈顿距离最近的坐标
                bdx = abs(box_best.xywh[0,0] - center_x)
                bdy = abs(box_best.xywh[0,1] - center_y)

                bcenter_x = box.xywh[0,0]
                bcenter_y = box.xywh[0,1]
                dx = abs(bcenter_x - center_x)
                dy = abs(bcenter_y - center_y)
                if (dx+dy) < (bdx+bdy):
                    box_best = box

            real_x = box_best.xywh[0,0]
            real_y = box_best.xywh[0,1]

            move_x = (real_x - center_x)
            move_y = (real_y - center_y)

            pyautogui.move(move_x,move_y)




    if cv2.waitKey(1) & 0xff == ord('q'):
        break

camera.stop()
cv2.destroyAllWindows()