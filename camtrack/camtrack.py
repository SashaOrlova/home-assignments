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

triang_params = TriangulationParameters(max_reprojection_error=1,
                                        min_triangulation_angle_deg=0.12,
                                        min_depth=0.07)


def build_and_triangulate_correspondences(corner_storage, intrinsic_mat, idx_1, mat_1, idx_2, mat_2):
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


def update_frames(frames, corners, builder, view_mats, cur_frame, mat):
    for frame in range(frames):
        if frame == cur_frame or view_mats[frame] is None:
            continue
        points, ids = build_and_triangulate_correspondences(corners, mat,
                                                            frame, view_mats[frame],
                                                            cur_frame, view_mats[cur_frame])
        if len(ids) != 0:
            builder.add_points(ids, points)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    points, ids = build_and_triangulate_correspondences(corner_storage, intrinsic_mat,
                                                        known_view_1[0], pose_to_view_mat3x4(known_view_1[1]),
                                                        known_view_2[0], pose_to_view_mat3x4(known_view_2[1]))
    if len(points) < 10:
        print("Слишком маленькое число точек, проверьте параметры запуска")
        exit(0)

    point_cloud_builder = PointCloudBuilder(ids, points)

    total_frames = len(corner_storage)
    view_mats = [None] * len(corner_storage)
    view_mats[known_view_1[0]] = pose_to_view_mat3x4(known_view_1[1])
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    counter = 0
    last_counter = counter
    while True:
        for cur_frame, corners in enumerate(corner_storage):
            if view_mats[cur_frame] is not None:
                continue

            _, comm1, comm2 = np.intersect1d(point_cloud_builder.ids.flatten(),
                                             corners.ids.flatten(),
                                             return_indices=True)
            try:
                inliers_len, mat = solve_ransac(point_cloud_builder.points[comm1], corners.points[comm2], intrinsic_mat)
                print(f'Обработан {cur_frame} из {total_frames}')
                if inliers_len > 0:
                    view_mats[cur_frame] = mat
                    counter += 1
                else:
                    continue
            except:
                print(f'Ошибка в {cur_frame} кадре')
                continue

            update_frames(total_frames, corner_storage, point_cloud_builder, view_mats, cur_frame, intrinsic_mat)

        if last_counter == counter:
            break
        else:
            last_counter = counter

    view_mats = np.array([x for x in view_mats if x is not None])
    if len(view_mats) == 0:
        print("Ничего не найдено, повторите с другими начальными параметрами")

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


def build_and_triangulate_correspondences(corner_storage, intrinsic_mat, idx_1, mat_1, idx_2, mat_2):
    correspondences = build_correspondences(corner_storage[idx_1], corner_storage[idx_2])
    if len(correspondences.ids) == 0:
        return [], []
    points, ids, _ = triangulate_correspondences(correspondences,
                                                 mat_1, mat_2,
                                                 intrinsic_mat,
                                                 triang_params)
    return points, ids


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
