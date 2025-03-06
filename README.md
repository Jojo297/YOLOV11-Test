# Installation

 1. Clone
 
    `git clone https://github.com/Jojo297/YOLOV5-Test.git`
    
 2. Install dependencies
	 
	 `pip install opencv-python ultralytics flask-socketio `
 
 3. Run
 
	 ```python your-name-file.py```

# clove ripeness detection

**Detection Image**

	```yolo predict model="runs/detect/train13/weights/best.pt" source=your-image.jpg```

**Detection from your webcam**

	yolo predict model="runs/detect/train13/weights/best.pt" source=0 show=True
