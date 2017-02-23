cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t

def bbox_overlaps(np.ndarray[DTYPE_t, ndim=2] anchors, np.ndarray[DTYPE_t, ndim=2] ground_truth):
    cdef unsigned int N = anchors.shape[0]
    cdef unsigned int K = ground_truth.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)

    cdef DTYPE_t box_area
    cdef DTYPE_t intersection_width, intersection_height

    cdef unsigned int k, n
    for k in range(K):
        box_area = (ground_truth[k, 2] - ground_truth[k, 0] + 1) * (ground_truth[k, 3] - ground_truth[k, 1] + 1)

        for n in range(N):
            intersection_width = min(anchors[n, 2], ground_truth[k, 2]) - max(anchors[n, 0], ground_truth[k, 0]) + 1

            if intersection_width > 0:
                intersection_height = min(anchors[n, 3], ground_truth[k, 3]) - max(anchors[n, 1], ground_truth[k, 1]) + 1

                if intersection_height > 0:
                    union_area = float((anchors[n, 2] - anchors[n, 0] + 1) * (anchors[n, 3] - anchors[n, 1] + 1) +
                                       box_area - intersection_width * intersection_height)

                    overlaps[n, k] = intersection_width * intersection_height / float(union_area)

        return overlaps
