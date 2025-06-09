from ultralytics import YOLO

class ObjectDetector:
    def __init__(self, model_path='yolov8s.pt', conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_objects(self, frame):
        results = self.model.predict(frame, conf=self.conf, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
        classes = results.boxes.cls.cpu().numpy() if results.boxes else []
        return boxes, classes

    def get_class_name(self, class_id):
        return self.model.names[int(class_id)]