#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'without_short_tracks'
]

'''
Идея подсмотренна из https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html
'''


import click
import cv2
import numpy as np
import pims

from _corners import FrameCorners, CornerStorage, StorageImpl
from _corners import dump, load, draw, without_short_tracks, create_cli


feature_params = dict(maxCorners=500,
                      qualityLevel=0.05,
                      minDistance=10,
                      blockSize=7)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


class _CornerStorageBuilder:
    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    frame = list(map(lambda t: (np.array(t) * 255.0).astype(np.uint8), frame_sequence))
    image_0 = frame[0]
    corners = cv2.goodFeaturesToTrack(image_0, **feature_params).squeeze(axis=1)
    points_end = len(corners)
    ids = np.arange(points_end)
    sizes = np.full(points_end, 10)
    builder.set_corners_at_frame(0, FrameCorners(ids, corners, sizes))

    for idx, image_1 in enumerate(frame[1:]):
        next = cv2.calcOpticalFlowPyrLK(image_0, image_1, corners, None, **lk_params)[0].squeeze()
        last = cv2.calcOpticalFlowPyrLK(image_1, image_0, next, None, **lk_params)[0].squeeze()
        mask = np.abs(corners - last).max(-1) < 0.2
        ids, corners, sizes = ids[mask], next[mask], sizes[mask]

        if len(corners) < 500:
            mask = np.ones_like(image_1, dtype=np.uint8)
            for x, y in corners:
                cv2.circle(mask, (x, y), 10, 0, -1)
            features = cv2.goodFeaturesToTrack(image_1, mask=mask*225, **feature_params)
            features = features.squeeze(axis=1) if features is not None else []
            for corner in features[:500 - len(corners)]:
                ids = np.concatenate([ids, [points_end]])
                points_end += 1
                corners = np.concatenate([corners, [corner]])
                sizes = np.concatenate([sizes, [10]])
        builder.set_corners_at_frame(idx, FrameCorners(ids, corners, sizes))
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.
    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter