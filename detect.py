from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO('best.pt')
    model.predict(source='tester.jpg', save=True, show=True)