import os
import math
import torch
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

# Configuration Parameters
class Config:
    # Paths
    pose_dir = "/mnt/HDD1/jl17001/dog_data/match_gps_ros_pose"  # Pose directory
    image_root_dir = "/mnt/HDD1/jl17001/dog_data"  # Image root directory
    output_dir = "/mnt/HDD1/jl17001/dog_data/pose_label"  # Output directory for label files

    # Categorization Thresholds
    CROWD_THRESHOLD = 5  # N = 5 people
    PERSON_CLOSE_BY_AREA_THRESHOLD = 40000  # pixels squared
    TURN_ANGLE_THRESHOLD = 20  # degrees
    ACTION_TARGET_MISMATCH_THRESHOLD = 45  # degrees
    CATEGORY_WINDOW = 0  # Previous and next 10 samples
    CROSS_THRESHOLD = 0  # pixels squared

    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model Configuration
    mask_rcnn_threshold = 0.5  # Confidence threshold for detections

cfg = Config()

def compute_angle(v1, v2):
    """Compute the angle in degrees between two vectors."""
    dot_prod = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_angle = np.clip(dot_prod / (norm_v1 * norm_v2), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def compute_movement_distance(pose_line):
    """
    Compute the movement distance as the norm of the (tx, ty, tz) vector.
    """
    tokens = pose_line.split(',')
    tx = float(tokens[1])
    ty = float(tokens[2])
    tz = float(tokens[3])
    return np.linalg.norm([tx, ty, tz])

def process_pose_files(cfg):
    """
    Process all pose files and categorize each data sample.
    """
    # Initialize Mask R-CNN model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(weights='DEFAULT')
    model.to(cfg.device)
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to Tensor
    ])

    # Load predefined pose files
    pose_files = [
        "match_gps_ros_pose23.txt",
        "match_gps_ros_pose24.txt",
        "match_gps_ros_pose25.txt",
        "match_gps_ros_pose26.txt",
        "match_gps_ros_pose27.txt",
        "match_gps_ros_pose28.txt",
        "match_gps_ros_pose29.txt",
        "match_gps_ros_pose31.txt",
        "match_gps_ros_pose33.txt",
    ]
    pose_paths = [os.path.join(cfg.pose_dir, f) for f in pose_files]

    # Ensure output directory exists
    os.makedirs(cfg.output_dir, exist_ok=True)

    for pose_path in tqdm(pose_paths, desc="Processing pose files"):
        print(f"\nProcessing pose file: {pose_path}")
        seq_idx = ''.join(filter(str.isdigit, os.path.basename(pose_path)))
        image_folder = os.path.join(cfg.image_root_dir, f'dog_nav_undistort_{seq_idx}')
        if not os.path.exists(image_folder):
            print(f"Warning: Image folder {image_folder} does not exist. Skipping this pose file.")
            continue

        with open(pose_path, 'r') as file:
            lines = file.readlines()

        if len(lines) % 2 != 0:
            print(f"Warning: Pose file {pose_path} has an odd number of lines. Skipping the last incomplete sample.")
            lines = lines[:-1]

        samples = []  # Each sample is a dict with keys: 'line1', 'line2', 'categories'

        # Reference GPS point for local coordinate conversion
        ref_lat, ref_lon, ref_alt = None, None, None

        for i in tqdm(range(0, len(lines), 2), desc=f"Processing samples in {seq_idx}"):
            gps_line = lines[i].strip()
            pose_line = lines[i+1].strip()

            # Store original lines
            sample = {
                'line1': gps_line,
                'line2': pose_line,
                'categories': {
                    'crowd': 0,
                    'person_close_by': 0,
                    'turn': 0,
                    'action_target_mismatch': 0,
                    'crossing': 0
                }
            }

            # Parse pose data
            pose_tokens = pose_line.split(',')
            image_name = f"forward_{int(pose_tokens[7]):04d}.jpg"

            # Load the image
            image_path = os.path.join(image_folder, image_name)
            if not os.path.exists(image_path):
                print(f"Warning: Image {image_path} does not exist. Skipping this sample.")
                continue
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error loading image {image_path}: {e}. Skipping this sample.")
                continue

            # Perform object detection
            image_tensor = transform(image).to(cfg.device)
            with torch.no_grad():
                outputs = model([image_tensor])[0]

            # Filter detections for 'person' class (COCO class 1) with scores >= threshold
            person_indices = [
                i for i, label in enumerate(outputs['labels'].cpu().numpy())
                if label == 1 and outputs['scores'][i] >= cfg.mask_rcnn_threshold
            ]
            num_persons = len(person_indices)

            # Category 1: Crowd
            if num_persons > cfg.CROWD_THRESHOLD:
                sample['categories']['crowd'] = 1

            # Category 2: Person Close By
            # Compute areas of bounding boxes
            if 'boxes' in outputs and len(person_indices) > 0:
                boxes = outputs['boxes'][person_indices]
                box_areas = []
                for box in boxes:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    box_areas.append(area)
                if len(box_areas) > 0 and max(box_areas) > cfg.PERSON_CLOSE_BY_AREA_THRESHOLD:
                    sample['categories']['person_close_by'] = 1

            # Category 5: Crossing
            # Detect 'traffic light' (10), 'traffic sign' (11), 'stop sign' (13)
            crossing_labels = [10, 11, 13]
            crossing_indices = [
                i for i, (label, score) in enumerate(zip(outputs['labels'].cpu().numpy(), outputs['scores'].cpu().numpy()))
                if label in crossing_labels and score >= cfg.mask_rcnn_threshold
            ]
            if len(crossing_indices) > 0:
                boxes = outputs['boxes'][crossing_indices]
                box_areas = []
                for box in boxes:
                    x1, y1, x2, y2 = box.cpu().numpy()
                    area = (x2 - x1) * (y2 - y1)
                    box_areas.append(area)
                if len(box_areas) > 0 and min(box_areas) > cfg.CROSS_THRESHOLD:
                    sample['categories']['crossing'] = 1

            samples.append(sample)

        # ----- New Filtering Step Starts Here -----
        # Filter out initial samples with movement distance < 0.1 meters
        print(f"\nFiltering initial samples with movement distance < 0.1m for trajectory {seq_idx}...")
        movement_distances = [compute_movement_distance(sample['line2']) for sample in samples]

        # Find the first index where movement distance >= 0.1m
        filter_idx = 0
        for idx, dist in enumerate(movement_distances):
            if dist >= 0.1:
                filter_idx = idx
                break
        else:
            # If no movement distance >= 0.1m, skip this trajectory
            print(f"All samples in {pose_path} have movement distance < 0.1m. Skipping trajectory {seq_idx}.")
            continue

        if filter_idx > 0:
            print(f"Filtered out {filter_idx} initial samples with movement distance < 0.1m from trajectory {seq_idx}.")
            samples = samples[filter_idx:]
        else:
            print(f"No initial samples to filter for trajectory {seq_idx}.")

        # ----- New Filtering Step Ends Here -----

        # After loading all samples, process pose-based categories ('turn' and 'action_target_mismatch')
        print(f"Processing pose-based categories for trajectory {seq_idx}...")

        num_samples = len(samples)
        if num_samples == 0:
            print(f"No samples to process for trajectory {seq_idx} after filtering. Skipping.")
            continue

        # Parse all poses to extract waypoints
        poses = []
        for sample in samples:
            pose_line = sample['line2']
            pose_tokens = pose_line.split(',')
            if len(pose_tokens) < 8:
                print(f"Warning: Pose line '{pose_line}' is malformed. Skipping this sample.")
                poses.append([0, 0, 0, 0, 0, 0])  # Placeholder for malformed pose
                continue
            try:
                tx = float(pose_tokens[1])
                ty = float(pose_tokens[2])
                tz = float(pose_tokens[3])
                rx = float(pose_tokens[4])
                ry = float(pose_tokens[5])
                rz = float(pose_tokens[6])
                poses.append([tx, ty, tz, rx, ry, rz])
            except ValueError:
                print(f"Error parsing pose line '{pose_line}'. Using zero transformation.")
                poses.append([0, 0, 0, 0, 0, 0])  # Placeholder for parsing error
        poses = np.array(poses)  # Shape: (num_samples, 6)

        # Compute transformation matrices for all poses
        pose_matrices = []
        for pose in poses:
            tx, ty, tz, rx, ry, rz = pose
            rotation = R.from_rotvec([rx, ry, rz])
            matrix = np.eye(4)
            matrix[:3, :3] = rotation.as_matrix()
            matrix[:3, 3] = [tx, ty, tz]
            pose_matrices.append(matrix)
        pose_matrices = np.array(pose_matrices)  # Shape: (num_samples, 4, 4)

        # Compute relative positions for each sample
        positions_2d = np.zeros((num_samples, 2))

        for idx in range(num_samples-1):
            current_pose_matrix = pose_matrices[idx]
            try:
                current_pose_inv = np.linalg.inv(current_pose_matrix)
            except np.linalg.LinAlgError:
                print(f"Warning: Singular matrix encountered at index {idx} in trajectory {seq_idx}. Using identity matrix.")
                current_pose_inv = np.eye(4)

            # Determine the target index (50th pose after current or last pose)
            target_idx = min(idx + 50, num_samples - 1)

            # Extract waypoints: next five poses after current
            waypoint_end_idx = min(idx + 5, num_samples - 1)
            waypoints = pose_matrices[idx + 1: waypoint_end_idx + 1]  # Shape: (num_waypoints, 4, 4)

            # Define target pose
            target_pose_matrix = pose_matrices[target_idx]

            # Transform waypoints and target to the current pose's coordinate frame
            transformed_waypoints = np.matmul(current_pose_inv, waypoints)  # Shape: (5, 4, 4)
            transformed_target = np.matmul(current_pose_inv, target_pose_matrix)  # Shape: (4, 4)

            # Extract 2D positions (assuming x and y are the first two coordinates)
            waypoints_2d = transformed_waypoints[:, :2, 3]  # Shape: (5, 2)
            target_2d = transformed_target[:2, 3]  # Shape: (2,)

            # Category 3: Turn
            # Compute the angle between the first waypoint vector and the difference between the fourth and fifth waypoint vectors
            last_vector = waypoints_2d[-1]
            current_orientation = np.array([-1, 0])
            angle_turn = compute_angle(current_orientation, last_vector)
            if angle_turn > cfg.TURN_ANGLE_THRESHOLD:
                samples[idx]['categories']['turn'] = 1

            # Category 4: Action-Target Mismatch
            # Compute the mean of waypoint differences
            if waypoints_2d.shape[0] >= 2:
                angle_mismatch = compute_angle(last_vector, target_2d)
                if angle_mismatch > cfg.ACTION_TARGET_MISMATCH_THRESHOLD:
                    samples[idx]['categories']['action_target_mismatch'] = 1

        # Initialize category lists
        categories = {
            'crowd': [0] * num_samples,
            'person_close_by': [0] * num_samples,
            'turn': [0] * num_samples,
            'action_target_mismatch': [0] * num_samples,
            'crossing': [0] * num_samples
        }

        for idx, sample in enumerate(samples):
            categories['crowd'][idx] = sample['categories']['crowd']
            categories['person_close_by'][idx] = sample['categories']['person_close_by']
            categories['turn'][idx] = sample['categories']['turn']
            categories['action_target_mismatch'][idx] = sample['categories']['action_target_mismatch']
            categories['crossing'][idx] = sample['categories']['crossing']

        # Expand categories by labeling previous and next 10 samples
        print("Expanding category labels with window...")
        for category, labels in categories.items():
            expanded_labels = labels.copy()
            for idx, label in enumerate(labels):
                if label:
                    start = max(idx - cfg.CATEGORY_WINDOW, 0)
                    end = min(idx + cfg.CATEGORY_WINDOW + 1, num_samples)
                    for i in range(start, end):
                        expanded_labels[i] = 1
            categories[category] = expanded_labels

        # Assign 'other' labels
        print("Assigning 'other' labels...")
        other_labels = []
        for idx in range(num_samples):
            if any(categories[cat][idx] for cat in ['crowd', 'person_close_by', 'turn', 'action_target_mismatch', 'crossing']):
                other_labels.append(0)
            else:
                other_labels.append(1)
        categories['other'] = other_labels

        # Prepare the final category lines
        final_categories = []
        for idx in range(num_samples):
            cat_values = [
                str(categories['crowd'][idx]),
                str(categories['person_close_by'][idx]),
                str(categories['turn'][idx]),
                str(categories['action_target_mismatch'][idx]),
                str(categories['crossing'][idx]),
                str(categories['other'][idx])
            ]
            cat_line = ','.join(cat_values)
            final_categories.append(cat_line)

        # Write to the output file for this trajectory
        output_file_path = os.path.join(cfg.output_dir, f"pose_label_{seq_idx}.txt")
        print(f"Writing categorized data to {output_file_path}...")
        with open(output_file_path, 'w') as f_out:
            for sample, cat_line in zip(samples, final_categories):
                f_out.write(sample['line1'] + '\n')
                f_out.write(sample['line2'] + '\n')
                f_out.write(cat_line + '\n')

        # Calculate summary statistics
        total_samples = num_samples
        category_counts = {
            'crowd': sum(categories['crowd']),
            'person_close_by': sum(categories['person_close_by']),
            'turn': sum(categories['turn']),
            'action_target_mismatch': sum(categories['action_target_mismatch']),
            'crossing': sum(categories['crossing']),
            'other': sum(categories['other'])
        }

        # Print summary for this trajectory
        print(f"Trajectory {seq_idx} Summary:")
        print(f"Total samples: {total_samples}")
        print(f"Crowd: {category_counts['crowd']}")
        print(f"Person Close By: {category_counts['person_close_by']}")
        print(f"Turn: {category_counts['turn']}")
        print(f"Action-Target Mismatch: {category_counts['action_target_mismatch']}")
        print(f"Crossing: {category_counts['crossing']}")
        print(f"Other: {category_counts['other']}")
        print("-" * 50)

    print("Data categorization and export completed successfully.")

if __name__ == "__main__":
    process_pose_files(cfg)
