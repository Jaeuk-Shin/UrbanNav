import cv2
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from carla_utils.tf import UE


def build_intrinsic_matrix(w, h, fov):
    f = .5 * w / np.tan(fov * np.pi / 360.0)
    cx, cy = .5 * w, .5 * h
    return np.array([
        [f, 0.,  cx],
        [0., f,  cy],
        [0., 0., 1.]
        ])


def load_camera_parameters(filepath):

    basis_change = UE[:3, :3]
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)['sensors']
    params = {}
    for sensor in data:
        if 'camera' in sensor['type']:
            e = sensor['spawn_point']       # extrinsic
            i = sensor['attributes']        # intrinsic

            orientation = Rotation.from_euler(
                'xyz',
                [e['roll'], e['pitch'], e['yaw']],
                degrees=True
            )

            # The following are only for CARLA!
            orientation = basis_change @ orientation.as_matrix() @ basis_change.T
            orientation = Rotation.from_matrix(orientation)
            position = np.array([e['y'], -e['z'], e['x']])

            K = build_intrinsic_matrix(
                w=i['image_size_x'],
                h=i['image_size_y'],
                fov=i['fov']
            )
            params[sensor['id']] = {
                'extrinsic': {'position': position, 'orientation': orientation},
                'intrinsic': K
            }
    return params


FRONT_THRESHOLD = 0.01
def get_image_coordinates(points, K):
    """
    points: numpy array of shape (# of points, 3); coordinates in camera frame
    K: intrinsic matrix
    """
    assert points.ndim == 2 and points.shape[-1] == 3
    points_img = K @ points.T
    x_img, y_img, z_img = points_img
    z_img += 1e-8       # to avoid division by 0
    u, v = x_img / z_img, y_img / z_img     # shape: (*,)

    valid = z_img > FRONT_THRESHOLD     # in front of the camera
    return u, v, valid





def define_virtual_camera(camera1, camera2, w, h, fov):
    p1 = camera1['extrinsic']['position']
    p2 = camera2['extrinsic']['position']
    o1 = camera1['extrinsic']['orientation']
    o2 = camera2['extrinsic']['orientation']

    key_rots = Rotation.concatenate([o1, o2])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    
    orientation = slerp(0.5)
    position = .5 * (p1 + p2)

    K = build_intrinsic_matrix(w, h, fov)
    K[0, 2] -= 150
    K[1, 2] -= 50

    return {
        'extrinsic': {'position': position, 'orientation': orientation},
        'intrinsic': K
    }


def compute_homography_matrix(source_cam_params, target_cam_params):
    R_source = source_cam_params['extrinsic']['orientation']
    R_target = target_cam_params['extrinsic']['orientation']
    R_rel = R_source.as_matrix() @ R_target.as_matrix().T
    
    K_source = source_cam_params['intrinsic']
    K_target = target_cam_params['intrinsic']
    H = K_source @ R_rel @ np.linalg.inv(K_target)

    return H


def warp_image(img, H, w, h):
    return cv2.warpPerspective(img, H, (w, h))


def test_image_rectification():
    filepath = os.path.join(os.path.dirname(__file__), 'sensors.json')
    
    
    params = load_camera_parameters(filepath)

    fl = params['flcam']
    fr = params['frcam']

    w, h, fov = 1280, 720, 120
    center = define_virtual_camera(fl, fr, w, h, fov)

    H_fl = compute_homography_matrix(fl, center)
    H_fr = compute_homography_matrix(fr, center)

    datadir = '/mnt/ssd3/carla_dataset/0001'

    filepath_fl = os.path.join(datadir, 'flcam', '0001.jpg')
    filepath_fr = os.path.join(datadir, 'frcam', '0001.jpg')
    img_fl = cv2.imread(str(filepath_fl))
    img_fr = cv2.imread(str(filepath_fr))

    fl2center = warp_image(img_fl, H_fl, w, h)
    fr2center = warp_image(img_fr, H_fr, w, h)

    unified = np.maximum(fl2center, fr2center)

    output_filepath = os.path.join(os.path.dirname(__file__), 'rectified_example.jpg')
    cv2.imwrite(output_filepath, unified, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


if __name__ == '__main__':
    test_image_rectification()