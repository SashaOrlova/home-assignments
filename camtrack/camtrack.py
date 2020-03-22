#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import cv2

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4,
    pose_to_view_mat3x4
)


def get_params():
    return TriangulationParameters(max_reprojection_error=0.1,
                                   min_triangulation_angle_deg=4,
                                   min_depth=4)


def build_and_get_correspondences(corner_storage, intrinsic_mat, idx_1, mat_1, idx_2, mat_2, triang_params):
    correspondences = build_correspondences(corner_storage[idx_1], corner_storage[idx_2])
    if len(correspondences.ids) == 0:
        return [], []
    points, ids, _ = triangulate_correspondences(correspondences,
                                                 mat_1, mat_2,
                                                 intrinsic_mat,
                                                 triang_params)
    return points, ids


def solve_ransac(object_points, image_points, intrinsic_mat):
    rv, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image_points, intrinsic_mat, None)
    inliers_len = len(inliers)
    if rv and inliers_len > 0:
        return inliers_len, rodrigues_and_translation_to_view_mat3x4(rvec, tvec)
    else:
        return 0, []


def update_frames(frames, corners, builder, tracked_mats, cur_frame, mat, triang_params):
    for frame in range(frames):
        if frame == cur_frame or tracked_mats[frame] is None:
            continue
        points, ids = build_and_get_correspondences(corners, mat,
                                                    frame, tracked_mats[frame],
                                                    cur_frame, tracked_mats[cur_frame], triang_params)
        if len(ids) != 0:
            builder.add_points(ids, points)


def init(corner_storage: CornerStorage,
         intrinsic_mat: np.ndarray,
         triangulation_parameters: TriangulationParameters):
    results = []
    for frame in range(1, len(corner_storage)):
        correspondences = build_correspondences(corner_storage[0], corner_storage[frame])
        if not len(correspondences[0]):
            continue

        e, mask = cv2.findEssentialMat(correspondences.points_1,
                                       correspondences.points_2,
                                       cameraMatrix=intrinsic_mat)
        rotation_matrix_1, rotation_matrix_2, t = cv2.decomposeEssentialMat(e)

        for R in [rotation_matrix_1, rotation_matrix_2]:
            try:
                results.append(triangulate_correspondences(correspondences,
                                                           np.eye(3, 4),
                                                           np.hstack([R, t]),
                                                           intrinsic_mat,
                                                           triangulation_parameters))
            except:
                print("Error")
                exit(0)

    if len(results) <= 0:
        print("Error")
        exit(0)

    min_cos = 2
    max_len_ps = -1
    res = []
    for r in results:
        if len(r[0]) > max_len_ps:
            max_len_ps = len(r[0])
            min_cos = r[2]
            res = r
        elif len(r[0]) == max_len_ps:
            if min_cos > r[2]:
                res = r
                min_cos = r[2]
    return res


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    triang_params = get_params()
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        points, ids, _ = init(corner_storage, intrinsic_mat, triang_params)
        print(f'Initialized with {len(points)} points.')
    else:
        points, ids = build_and_get_correspondences(corner_storage, intrinsic_mat,
                                                    known_view_1[0], pose_to_view_mat3x4(known_view_1[1]),
                                                    known_view_2[0], pose_to_view_mat3x4(known_view_2[1]),
                                                    triang_params)

    if len(points) < 7:
        print("Слишком маленькое число точек, проверьте параметры запуска")
        exit(0)

    point_cloud_builder = PointCloudBuilder(ids, points)

    total_frames = len(corner_storage)
    tracked_mats = [None] * len(corner_storage)

    for cur_frame, corners in enumerate(corner_storage):
        if tracked_mats[cur_frame] is not None:
            continue

        _, comm1, comm2 = np.intersect1d(point_cloud_builder.ids.flatten(),
                                         corners.ids.flatten(),
                                         return_indices=True)
        try:
            inliers_len, mat = solve_ransac(point_cloud_builder.points[comm1], corners.points[comm2], intrinsic_mat)
            print(f'Обработан {cur_frame} из {total_frames} с {inliers_len} инлайнерами')
            if inliers_len > 0:
                tracked_mats[cur_frame] = mat
            else:
                continue
        except:
            print(f'Ошибка в {cur_frame} кадре')
            continue

        update_frames(total_frames, corner_storage, point_cloud_builder, tracked_mats, cur_frame, intrinsic_mat,
                      triang_params)

    tracked_mats = np.array([x for x in tracked_mats if x is not None])
    if len(tracked_mats) == 0:
        print("Ничего не найдено, повторите с другими начальными параметрами")

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        tracked_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, tracked_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
