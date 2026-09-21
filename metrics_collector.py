import time
from collections import deque
import numpy as np

class MetricsCollector:
    def __init__(self, window_size=120):
        self.window_size = window_size
        
        self.inference_times = deque(maxlen=window_size)
        self.pipeline_times = deque(maxlen=window_size)
        self.frame_times = deque(maxlen=window_size)
        
        self.last_frame_time = None
        
    def add_inference_time(self, dt):
        self.inference_times.append(dt)
        
    def add_pipeline_time(self, dt):
        self.pipeline_times.append(dt)
        
    def mark_frame(self):
        current_time = time.perf_counter()
        if self.last_frame_time is not None:
            self.frame_times.append(current_time - self.last_frame_time)
        self.last_frame_time = current_time
        
    def _get_stats(self, deq):
        if not deq:
            return 0.0, 0.0, 0.0
        arr = np.array(deq) * 1000.0  # Convert to ms
        return np.mean(arr), np.median(arr), np.percentile(arr, 95)
        
    def get_inference_stats(self):
        """Returns (Mean ms, P50 ms, P95 ms) for inference latency"""
        return self._get_stats(self.inference_times)
        
    def get_pipeline_stats(self):
        """Returns (Mean ms, P50 ms, P95 ms) for end-to-end pipeline latency"""
        return self._get_stats(self.pipeline_times)
        
    def get_fps(self):
        """Returns (Mean FPS, P50 FPS, P5 FPS) - Note P5 FPS is equivalent to P95 frame time"""
        if not self.frame_times:
            return 0.0, 0.0, 0.0
        arr = np.array(self.frame_times)
        mean_time = np.mean(arr)
        p50_time = np.median(arr)
        p95_time = np.percentile(arr, 95)
        
        # Avoid division by zero
        if mean_time == 0: return 0.0, 0.0, 0.0
        
        return 1.0 / mean_time, 1.0 / p50_time, 1.0 / p95_time
