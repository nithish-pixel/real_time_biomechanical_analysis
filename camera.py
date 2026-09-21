import cv2
import threading

class ThreadedCamera:
    """
    Decouples OpenCV frame capture from the main processing loop
    to ensure the capture buffer doesn't build up and introduce latency.
    """
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src)
        
        # Configure camera for high performance if possible (might not be supported on all webcams)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.capture.set(cv2.CAP_PROP_FPS, 60)
        
        # Check if camera opened successfully
        if not self.capture.isOpened():
            raise ValueError(f"Unable to open camera source: {src}")
            
        self.status, self.frame = self.capture.read()
        self.started = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self
        
    def update(self):
        while self.started:
            status, frame = self.capture.read()
            with self.lock:
                self.status = status
                if status:
                    self.frame = frame
                    
    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.status, self.frame.copy()
            return self.status, None
            
    def stop(self):
        self.started = False
        if self.thread is not None:
            self.thread.join()
        self.capture.release()
