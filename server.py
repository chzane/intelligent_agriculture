# 奶牛检测

import cv2
from ultralytics import YOLO
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# YOLOv8 模型
model = YOLO('models/best.pt')

# 每个类别的数量
global_class_counts = {name: 0 for name in model.names.values()}

def generate_frames():
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("读取帧失败")
            break
 
        results = model(frame)

        local_class_counts = {name:0 for name in model.names.values()}

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框坐标
                confidence = box.conf[0]  # 置信度
                cls = int(box.cls[0])  # 类别
                label = model.names[cls]  # 类别名称

                local_class_counts[label] += 1

                # 绘制检测框和标签
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        global global_class_counts
        global_class_counts = local_class_counts

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # 视频流
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/class_counts')
def class_counts():
    # 为前端页面返回计数
    return jsonify(global_class_counts)

if __name__ == '__main__':
    app.run(debug=False, port=1024)
