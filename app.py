from ultralytics import YOLO

model = YOLO('/Users/gergelyfazekas/Documents/python_projects/bar_chart/runs/detect/train/weights/best.pt') 
model.predict(source='fig.png', save=True, hide_labels=True, hide_conf=True)


