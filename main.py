import cv2
import time
from camera import ThreadedCamera
from pose_tracker import PoseTracker
from biomechanics import BiomechanicsEngine
from metrics_collector import MetricsCollector
from constants import POSE_CONNECTIONS

def draw_hud(frame, metrics_collector, angles_results, landmarks_2d):
    """Overlays the HUD on the frame."""
    h, w, _ = frame.shape
    
    # Draw connections
    if landmarks_2d:
        for connection in POSE_CONNECTIONS:
            start_idx = connection[0]
            end_idx = connection[1]
            if start_idx in landmarks_2d and end_idx in landmarks_2d:
                p1 = landmarks_2d[start_idx]
                p2 = landmarks_2d[end_idx]
                
                # Check visibility for drawing (use 0.5 as drawing threshold, logic uses 0.65)
                if p1['visibility'] > 0.5 and p2['visibility'] > 0.5:
                    x1, y1 = int(p1['x'] * w), int(p1['y'] * h)
                    x2, y2 = int(p2['x'] * w), int(p2['y'] * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    
        # Draw points
        for idx, lm in landmarks_2d.items():
            if lm['visibility'] > 0.5:
                x, y = int(lm['x'] * w), int(lm['y'] * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # Performance Stats
    mean_fps, p50_fps, p95_fps = metrics_collector.get_fps()
    mean_inf, p50_inf, p95_inf = metrics_collector.get_inference_stats()
    mean_pipe, p50_pipe, p95_pipe = metrics_collector.get_pipeline_stats()
    
    cv2.rectangle(frame, (10, 10), (400, 140), (0, 0, 0), -1)
    
    cv2.putText(frame, f"FPS: Mean={mean_fps:.1f} P50={p50_fps:.1f}", (20, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Inference Latency: P50={p50_inf:.1f}ms P95={p95_inf:.1f}ms", (20, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"E2E Latency: P50={p50_pipe:.1f}ms P95={p95_pipe:.1f}ms", (20, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Angle Stats
    y_offset = 170
    cv2.rectangle(frame, (10, y_offset - 20), (450, h - 10), (0, 0, 0), -1)
    
    joints_to_show = [
        ('Elbow Flexion', 'elbow_flexion'),
        ('Knee Flexion', 'knee_flexion'),
        ('Hip Flexion', 'hip_flexion'),
        ('Shoulder Flex', 'shoulder_flexion'),
        ('Shoulder Abd', 'shoulder_abduction'),
        ('Ankle Dorsi', 'ankle_dorsiflexion')
    ]
    
    for label, key in joints_to_show:
        l_key = f'left_{key}'
        r_key = f'right_{key}'
        
        l_str = "---"
        l_color = (0, 0, 255) # Red for unreliable
        if l_key in angles_results:
            res = angles_results[l_key]
            l_str = f"{res['value']:.1f}"
            if res['reliable']: l_color = (0, 255, 0)
            
        r_str = "---"
        r_color = (0, 0, 255)
        if r_key in angles_results:
            res = angles_results[r_key]
            r_str = f"{res['value']:.1f}"
            if res['reliable']: r_color = (0, 255, 0)
            
        cv2.putText(frame, f"{label}:", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.putText(frame, f"L:{l_str}", (220, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, l_color, 1)
        cv2.putText(frame, f"R:{r_str}", (320, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_color, 1)
        
        y_offset += 30

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Real-Time Biomechanical Analysis")
    parser.add_argument("--source", default="0", help="Camera index (0, 1, ...) or video file path")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source

    try:
        camera = ThreadedCamera(src).start()
    except ValueError as e:
        print(f"\n[Camera Initialization Error]:\n{e}\n")
        return

    tracker = PoseTracker()
    engine = BiomechanicsEngine()
    metrics = MetricsCollector()
    
    print("Starting Biomechanical Analysis System...")
    print("Press 'q' to quit.")
    
    while True:
        pipeline_start = time.perf_counter()
        status, frame = camera.read()
        
        if not status or frame is None:
            time.sleep(0.01)
            continue
            
        # Mediapipe expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Pose Tracking (Inference + Filtering)
        tracker_res = tracker.process(frame_rgb)
        metrics.add_inference_time(tracker_res['inference_time'])
        
        # 2. Biomechanics (Angle Calculation)
        angles_results = engine.calculate_angles(tracker_res['world_landmarks'])
        
        # 3. Visualization
        draw_hud(frame, metrics, angles_results, tracker_res['landmarks_2d'])
        
        # Show image
        cv2.imshow('Real-Time Biomechanical Analysis', frame)
        
        # 4. Metrics
        metrics.add_pipeline_time(time.perf_counter() - pipeline_start)
        metrics.mark_frame()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    camera.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
