import time
import os
import urllib.request
import numpy as np
import mediapipe as mp
from app.filter import OneEuroFilter3D

class PoseTracker:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.model_path = "pose_landmarker_lite.task"
        if not os.path.exists(self.model_path):
            print("Downloading MediaPipe Pose Landmarker model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                self.model_path
            )
            
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        self.options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.landmarker = PoseLandmarker.create_from_options(self.options)
        
        self.filters = {} # Map landmark index to OneEuroFilter3D
        self.t0 = time.time()
        
    def process(self, frame_rgb, timestamp_ms):
        """
        Process the RGB frame, track pose, and return filtered world landmarks 
        and raw 2D landmarks for drawing.
        """
        start_time = time.perf_counter()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            results = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as e:
            # Sometime timestamp error, fallback
            results = None
            
        inference_time = time.perf_counter() - start_time
        
        current_time = time.time() - self.t0
        filtered_world_landmarks = {}
        landmarks_2d = {}
        visibility_scores = {}
        
        if results and results.pose_world_landmarks and len(results.pose_world_landmarks) > 0:
            # We take the first person
            world_lms = results.pose_world_landmarks[0]
            for idx, lm in enumerate(world_lms):
                # mp.tasks.vision returns visibility/presence in some cases as well
                vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else getattr(lm, 'presence', 1.0)
                visibility_scores[idx] = vis
                # Filter only if reasonably visible
                if vis > 0.65:
                    if idx not in self.filters:
                        self.filters[idx] = OneEuroFilter3D(current_time, lm.x, lm.y, lm.z, min_cutoff=0.1, beta=0.5)
                    
                    fx, fy, fz = self.filters[idx](current_time, lm.x, lm.y, lm.z)
                    filtered_world_landmarks[idx] = {'x': fx, 'y': fy, 'z': fz, 'visibility': vis}
                else:
                    filtered_world_landmarks[idx] = {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': vis}
                    # Reset filter for this joint if tracking is lost
                    if idx in self.filters:
                        del self.filters[idx]
                        
        if results and results.pose_landmarks and len(results.pose_landmarks) > 0:
            lms = results.pose_landmarks[0]
            for idx, lm in enumerate(lms):
                vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else getattr(lm, 'presence', 1.0)
                landmarks_2d[idx] = {'x': lm.x, 'y': lm.y, 'visibility': vis}
                
        return {
            'world_landmarks': filtered_world_landmarks,
            'landmarks_2d': landmarks_2d,
            'visibility': visibility_scores,
            'inference_time': inference_time
        }
