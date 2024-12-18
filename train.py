# 模型训练代码

from ultralytics import YOLO

model = YOLO('models/yolo11n.pt')

# 默认训练50步
model.train(data = './train.yaml', epochs = 40)

model.val()