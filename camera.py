import cv2
import threading
import os

class ThreadedCamera:
    """
    Decouples OpenCV frame capture from the main processing loop
    to ensure the capture buffer doesn't build up and introduce latency.
    """
    def __init__(self, src=0):
        self.src = src
        self.is_video_file = isinstance(src, str) and os.path.exists(src)

        if isinstance(src, int):
            # On Windows, try DirectShow first as MSMF often fails
            self.capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not self.capture.isOpened():
                self.capture = cv2.VideoCapture(src)
        else:
            self.capture = cv2.VideoCapture(src)
        
        # Check if camera/source opened successfully
        if not self.capture.isOpened():
            if isinstance(src, int):
                raise ValueError(
                    f"Unable to open camera index {src}. No active webcam was detected on your system.\n"
                    f"  - Please make sure your webcam is plugged in and not disabled in Windows settings.\n"
                    f"  - Alternatively, you can run the analysis on a video file using:\n"
                    f"      python main.py --source path/to/video.mp4"
                )
            else:
                raise ValueError(f"Unable to open video source: '{src}'")

        if not self.is_video_file:
            # Configure camera for high performance if supported
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.capture.set(cv2.CAP_PROP_FPS, 60)
            
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
            if not status and self.is_video_file:
                # Loop video file
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
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
