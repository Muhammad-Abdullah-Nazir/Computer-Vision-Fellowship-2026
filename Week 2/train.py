from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=r'C:\Users\SS\OneDrive\Desktop\University\Internship\Week 2\Bottle-Detection\dataset\bottle-8\data.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        workers=2,
        name='bottle_v1',
        project='runs'
    )

    print("Training complete!")