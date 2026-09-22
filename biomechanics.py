import numpy as np
from constants import PoseLandmark

def get_vector(landmarks, p1, p2):
    """Returns vector from p1 to p2."""
    return np.array([
        landmarks[p2]['x'] - landmarks[p1]['x'],
        landmarks[p2]['y'] - landmarks[p1]['y'],
        landmarks[p2]['z'] - landmarks[p1]['z']
    ])

def get_angle(v1, v2):
    """Returns the angle in degrees between two vectors."""
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    dot_prod = np.dot(v1, v2) / (norm_v1 * norm_v2)
    # Clip to avoid numerical instability
    dot_prod = np.clip(dot_prod, -1.0, 1.0)
    return np.degrees(np.arccos(dot_prod))

class BiomechanicsEngine:
    def __init__(self):
        self.min_visibility = 0.65

    def check_visibility(self, landmarks, indices):
        """Returns True if all required landmarks are sufficiently visible."""
        if not landmarks:
            return False
        for idx in indices:
            if idx not in landmarks or landmarks[idx]['visibility'] < self.min_visibility:
                return False
        return True

    def calculate_angles(self, world_landmarks):
        """Calculates goniometric angles based on 3D world landmarks."""
        results = {}
        if not world_landmarks:
            return results

        # Construct local coordinate system
        # Up (Y axis) = Midpoint of Hips to Midpoint of Shoulders
        if self.check_visibility(world_landmarks, [PoseLandmark.LEFT_HIP.value, PoseLandmark.RIGHT_HIP.value, PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value]):
            mid_hip = np.array([
                (world_landmarks[PoseLandmark.LEFT_HIP.value]['x'] + world_landmarks[PoseLandmark.RIGHT_HIP.value]['x']) / 2,
                (world_landmarks[PoseLandmark.LEFT_HIP.value]['y'] + world_landmarks[PoseLandmark.RIGHT_HIP.value]['y']) / 2,
                (world_landmarks[PoseLandmark.LEFT_HIP.value]['z'] + world_landmarks[PoseLandmark.RIGHT_HIP.value]['z']) / 2
            ])
            mid_shoulder = np.array([
                (world_landmarks[PoseLandmark.LEFT_SHOULDER.value]['x'] + world_landmarks[PoseLandmark.RIGHT_SHOULDER.value]['x']) / 2,
                (world_landmarks[PoseLandmark.LEFT_SHOULDER.value]['y'] + world_landmarks[PoseLandmark.RIGHT_SHOULDER.value]['y']) / 2,
                (world_landmarks[PoseLandmark.LEFT_SHOULDER.value]['z'] + world_landmarks[PoseLandmark.RIGHT_SHOULDER.value]['z']) / 2
            ])
            
            # Mediapipe Y axis goes down, so mid_shoulder to mid_hip points down.
            # To make an 'up' vector, we go mid_hip to mid_shoulder, or negative Y in camera space.
            up_vec = mid_shoulder - mid_hip
            up_vec = up_vec / np.linalg.norm(up_vec)
            
            # Right (X axis) = Left Hip to Right Hip
            right_vec = get_vector(world_landmarks, PoseLandmark.LEFT_HIP.value, PoseLandmark.RIGHT_HIP.value)
            right_vec = right_vec / np.linalg.norm(right_vec)
            
            # Forward (Z axis) = Up x Right
            forward_vec = np.cross(up_vec, right_vec)
            forward_vec = forward_vec / np.linalg.norm(forward_vec)
            
            # Re-orthogonalize up vector to ensure perfectly orthogonal basis
            up_vec = np.cross(right_vec, forward_vec)
            
            # Define planes (normals)
            sagittal_normal = right_vec
            coronal_normal = forward_vec
            
            results['body_orientation'] = True
        else:
            results['body_orientation'] = False
            up_vec = np.array([0, -1, 0])
            right_vec = np.array([1, 0, 0])
            forward_vec = np.array([0, 0, 1])
            sagittal_normal = right_vec
            coronal_normal = forward_vec

        # Helper to project a vector onto a plane defined by its normal
        def project_onto_plane(vec, normal):
            dist = np.dot(vec, normal)
            return vec - dist * normal

        # --- Elbow Flexion/Extension ---
        for side, sh, el, wr in [
            ('left', PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.LEFT_ELBOW.value, PoseLandmark.LEFT_WRIST.value),
            ('right', PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_ELBOW.value, PoseLandmark.RIGHT_WRIST.value)
        ]:
            if self.check_visibility(world_landmarks, [sh, el, wr]):
                v1 = get_vector(world_landmarks, el, sh)
                v2 = get_vector(world_landmarks, el, wr)
                angle = get_angle(v1, v2)
                # 0 at full extension, flexed to 150
                flexion = 180.0 - angle
                results[f'{side}_elbow_flexion'] = {'value': flexion, 'reliable': True}
            else:
                results[f'{side}_elbow_flexion'] = {'value': 0.0, 'reliable': False}

        # --- Knee Flexion/Extension ---
        for side, hi, kn, an in [
            ('left', PoseLandmark.LEFT_HIP.value, PoseLandmark.LEFT_KNEE.value, PoseLandmark.LEFT_ANKLE.value),
            ('right', PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value)
        ]:
            if self.check_visibility(world_landmarks, [hi, kn, an]):
                v1 = get_vector(world_landmarks, kn, hi)
                v2 = get_vector(world_landmarks, kn, an)
                angle = get_angle(v1, v2)
                # 0 at full extension, flexed to 135
                flexion = 180.0 - angle
                results[f'{side}_knee_flexion'] = {'value': flexion, 'reliable': True}
            else:
                results[f'{side}_knee_flexion'] = {'value': 0.0, 'reliable': False}

        # --- Hip Flexion/Extension ---
        for side, hi, kn in [
            ('left', PoseLandmark.LEFT_HIP.value, PoseLandmark.LEFT_KNEE.value),
            ('right', PoseLandmark.RIGHT_HIP.value, PoseLandmark.RIGHT_KNEE.value)
        ]:
            if self.check_visibility(world_landmarks, [hi, kn]) and results['body_orientation']:
                # Vector hip to knee
                hip_to_knee = get_vector(world_landmarks, hi, kn)
                # Project onto sagittal plane
                proj_vec = project_onto_plane(hip_to_knee, sagittal_normal)
                # Compare to down vector (-up_vec)
                down_vec = -up_vec
                angle = get_angle(proj_vec, down_vec)
                
                # Determine sign (forward swing = flexion, backward = extension)
                # Dot product with forward vector
                if np.dot(proj_vec, forward_vec) > 0:
                    # Flexion (forward)
                    val = angle
                else:
                    # Extension (backward), usually denoted as negative or specific extension value
                    val = -angle
                    
                results[f'{side}_hip_flexion'] = {'value': val, 'reliable': True}
            else:
                results[f'{side}_hip_flexion'] = {'value': 0.0, 'reliable': False}

        # --- Shoulder Flexion & Abduction ---
        for side, sh, el in [
            ('left', PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.LEFT_ELBOW.value),
            ('right', PoseLandmark.RIGHT_SHOULDER.value, PoseLandmark.RIGHT_ELBOW.value)
        ]:
            if self.check_visibility(world_landmarks, [sh, el]) and results['body_orientation']:
                # Vector shoulder to elbow
                sh_to_el = get_vector(world_landmarks, sh, el)
                down_vec = -up_vec
                
                # Abduction: Project onto Coronal Plane
                proj_coronal = project_onto_plane(sh_to_el, coronal_normal)
                abduction = get_angle(proj_coronal, down_vec)
                
                # Flexion: Project onto Sagittal Plane
                proj_sagittal = project_onto_plane(sh_to_el, sagittal_normal)
                flexion = get_angle(proj_sagittal, down_vec)
                
                # Correct sign for Flexion (forward is positive)
                if np.dot(proj_sagittal, forward_vec) < 0:
                    flexion = -flexion
                
                results[f'{side}_shoulder_flexion'] = {'value': flexion, 'reliable': True}
                results[f'{side}_shoulder_abduction'] = {'value': abduction, 'reliable': True}
            else:
                results[f'{side}_shoulder_flexion'] = {'value': 0.0, 'reliable': False}
                results[f'{side}_shoulder_abduction'] = {'value': 0.0, 'reliable': False}

        # --- Ankle Dorsiflexion/Plantarflexion ---
        for side, kn, an, fi in [
            ('left', PoseLandmark.LEFT_KNEE.value, PoseLandmark.LEFT_ANKLE.value, PoseLandmark.LEFT_FOOT_INDEX.value),
            ('right', PoseLandmark.RIGHT_KNEE.value, PoseLandmark.RIGHT_ANKLE.value, PoseLandmark.RIGHT_FOOT_INDEX.value)
        ]:
            if self.check_visibility(world_landmarks, [kn, an, fi]):
                # shank = knee to ankle
                v1 = get_vector(world_landmarks, kn, an)
                # foot = ankle to foot_index
                v2 = get_vector(world_landmarks, an, fi)
                alpha = get_angle(v1, v2)
                
                # Dorsiflexion < 90, Plantarflexion > 90. 
                # Output positive for dorsi, negative for plantar, or separate them. Let's output a single 'dorsiflexion' where positive is dorsiflexion, negative is plantarflexion.
                # Standard says neutral is 0. 90 deg = 0.
                dorsiflexion = 90.0 - alpha
                
                results[f'{side}_ankle_dorsiflexion'] = {'value': dorsiflexion, 'reliable': True}
            else:
                results[f'{side}_ankle_dorsiflexion'] = {'value': 0.0, 'reliable': False}

        return results
