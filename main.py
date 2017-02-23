import numpy as np

import cv2

import tensorflow as tf
import tensorflow.contrib.slim as slim

from cython_bbox import bbox_overlaps

NUM_CLASSES = 20
BATCH_SIZE = 128

FEAT_STRIDE = [16, ]
BASE_SIZE = 16
ANCHOR_SCALES = np.array([8, 16, 32])

RPN_NMS_THRESH = 0.7
RPN_PRE_NMS_TOP_N = 12000
RPN_POST_NMS_TOP_N = 2000

RPN_POSITIVE_OVERLAP = 0.7
RPN_NEGATIVE_OVERLAP = 0.3

RPN_BATCH_SIZE = 256
RPN_FG_FRACTION = 0.5


def vgg16_conv(inputs):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        net = slim.max_pool2d(net, [2, 2], scope='pool3')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')

        return net


def rpn_layers(inputs):
    with slim.arg_scope([slim.conv2d],
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        rpn_conv = slim.conv2d(inputs, 512, [3, 3],
                               activation_fn=tf.nn.relu,
                               scope='rpn_conv')

        rpn_cls_score = slim.conv2d(rpn_conv, 18, [1, 1],
                                    activation_fn=None,
                                    padding='VALID',
                                    scope='rpn_cls_score')
        shape = rpn_cls_score.shape()
        rpn_cls_score_reshaped = tf.reshape(rpn_cls_score, [BATCH_SIZE, shape[1], shape[2] * 9, 2])
        rpn_cls_prob_reshaped = slim.softmax(rpn_cls_score_reshaped, scope='rpn_cls_prob')
        rpn_cls_prob = tf.reshape(rpn_cls_prob_reshaped, [BATCH_SIZE, shape[1], shape[2], 18])

        rpn_bbox_pred = slim.conv2d(rpn_conv, 36, [1, 1],
                                    activation_fn=None,
                                    padding='VALID',
                                    scope='rpn_bbox_pred')

    return rpn_cls_prob, rpn_bbox_pred


def scale_anchors(base_anchor, scales):
    # Find the center of the base_anchor
    width = base_anchor[2] - base_anchor[0] + 1
    height = base_anchor[3] - base_anchor[1] + 1
    center_x = base_anchor[0] + (width - 1) / 2
    center_y = base_anchor[1] + (height - 1) / 2

    # Scale
    widths = width * scales
    heights = height * scales

    # Generate anchors
    widths = widths[:, np.newaxis]
    heights = heights[:, np.newaxis]
    anchors = np.hstack((center_x - (widths - 1) / 2,
                         center_y - (heights - 1) / 2,
                         center_x + (widths - 1) / 2,
                         center_y + (heights - 1) / 2))

    return anchors


def generate_anchors(im_height, im_width, feat_stride, anchor_scales):
    base_size = BASE_SIZE
    ratios = np.array([0.5, 1, 2])
    scales = anchor_scales

    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])

    # Find the center of the base_anchor
    width = base_anchor[2] - base_anchor[0] + 1
    height = base_anchor[3] - base_anchor[1] + 1
    center_x = base_anchor[0] + (width - 1) / 2
    center_y = base_anchor[1] + (height - 1) / 2

    # Generate heights and widths of base anchors with each ratio
    size = height * width
    size_ratios = size * ratios
    widths = np.round(np.sqrt(size_ratios))
    heights = np.round(widths * ratios)

    # Generate base anchor list
    heights = heights[:, np.newaxis]
    widths = widths[:, np.newaxis]
    base_anchors = np.hstack((center_x - (widths - 1) / 2,
                              center_y - (heights - 1) / 2,
                              center_x + (widths - 1) / 2,
                              center_y + (heights - 1) / 2))

    # Scale the anchors
    anchors = np.vstack([scale_anchors(base_anchors[i, :], scales) for i in range(base_anchors.shape[0])])

    # Generate shift coordinates
    shift_x = np.arange(0, im_width) * feat_stride
    shift_y = np.arange(0, im_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()

    # Shift the anchors
    A = anchors.shape[0]
    K = shifts.shape[0]

    anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)

    return anchors


def bbox_finetune(anchors, rpn_bbox, im_shape):
    if anchors.shape[0] == 0:
        return np.zeros((0, rpn_bbox.shape[1]), dtype=rpn_bbox.dtype)

    anchors = anchors.astype(rpn_bbox.dtype, copy=False)

    widths = anchors[:, 2] - anchors[:, 0] + 1
    heights = anchors[:, 3] - anchors[:, 1] + 1
    center_x = anchors[:, 0] + widths / 2
    center_y = anchors[:, 1] + heights / 2

    dx = rpn_bbox[:, 0::4]
    dy = rpn_bbox[:, 1::4]
    dw = rpn_bbox[:, 2::4]
    dh = rpn_bbox[:, 3::4]

    new_center_x = dx * widths[:, np.newaxis] + center_x[:, np.newaxis]
    new_center_y = dy * heights[:, np.newaxis] + center_y[:, np.newaxis]
    new_widths = np.exp(dw) * widths[:, np.newaxis]
    new_heights = np.exp(dh) * heights[:, np.newaxis]

    new_bbox = np.hstack((new_center_x - new_widths / 2,
                          new_center_y - new_heights / 2,
                          new_center_x + new_widths / 2,
                          new_center_y + new_heights / 2))

    # Clip bbox into the boundaries
    new_bbox[:, 0::4] = np.maximum(np.minimum(new_bbox[:, 0::4], im_shape[1] - 1), 0)
    new_bbox[:, 1::4] = np.maximum(np.minimum(new_bbox[:, 1::4], im_shape[0] - 1), 0)
    new_bbox[:, 2::4] = np.maximum(np.minimum(new_bbox[:, 2::4], im_shape[1] - 1), 0)
    new_bbox[:, 3::4] = np.maximum(np.minimum(new_bbox[:, 3::4], im_shape[1] - 1), 0)

    return new_bbox


def nms(mat, scores, threshold):
    x1 = mat[:, 0]
    y1 = mat[:, 1]
    x2 = mat[:, 2]
    y2 = mat[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        new_x1 = np.maximum(x1[i], x1[order[1:]])
        new_y1 = np.maximum(y1[i], y1[order[1:]])
        new_x2 = np.minimum(x2[i], x2[order[1:]])
        new_y2 = np.minimum(y2[i], y2[order[1:]])

        new_width = np.maximum(0.0, new_x2 - new_x1 + 1)
        new_height = np.maximum(0.0, new_y2 - new_y1 + 1)

        intersection = new_width * new_height
        overlap = intersection / (areas[i] + areas[order[1:]] - intersection)

        index = np.where(overlap <= threshold)[0]

        order = order[index + 1]

    return keep


def proposal_layer(rpn_cls_prob, rpn_bbox_pred, im_info, feat_stride, anchors, scales):
    rpn_num_thresh = RPN_NMS_THRESH
    rpn_pre_nms_top_n = RPN_PRE_NMS_TOP_N
    rpn_post_nms_top_n = RPN_POST_NMS_TOP_N

    num_anchors = scales.shape[0] * 3

    scores = rpn_cls_prob[:, :, :, num_anchors:].reshape((-1, 1))
    bbox = rpn_bbox_pred.reshape((-1, 4))

    proposals = bbox_finetune(anchors, bbox, im_info[0][:2])

    # Pick top-rated bbox
    order = scores.ravel().argsort()[::-1][:rpn_pre_nms_top_n]
    proposals = proposals[order, :]
    scores = scores[order]

    # Non-maximal Suppression
    keep = nms(proposals, scores, rpn_num_thresh)

    # Pick top-rated again
    proposals = proposals[keep, :]
    scores = scores[keep]

    batch_index = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_index, proposals.astype(np.float32, copy=False)))

    return blob, scores


def anchor_target_layer(rpn_cls_prob, ground_truth, im_info, feat_stride, original_anchors, scales):
    A = scales.shape[0] * 3
    K = original_anchors.shape[0] / A

    rpn_positive_overlap = RPN_POSITIVE_OVERLAP
    rpn_negative_overlap = RPN_NEGATIVE_OVERLAP

    rpn_fg_fraction = RPN_FG_FRACTION

    allowed_border_width = 0

    height, width = rpn_cls_prob.shape[1:3]

    indices_within_border = np.where(
        (original_anchors[:, 0] >= -allowed_border_width) and
        (original_anchors[:, 1] >= -allowed_border_width) and
        (original_anchors[:, 2] < im_info[0][1] + allowed_border_width) and
        (original_anchors[:, 3] < im_info[0][0] + allowed_border_width)
    )[0]

    anchors = original_anchors[indices_within_border, :]

    labels = np.empty((len(indices_within_border), ), dtype=np.float32)
    labels.fill(-1)  # 1 for positive, 0 for negative, -1 for ambiguous samples

    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(ground_truth, dtype=np.float)
    )  # A N*K matrix of IoU

    overlaps_max = overlaps[np.arange(len(indices_within_border)), overlaps.argmax(axis=1)]
    overlaps_groundtruth_max = overlaps[overlaps.argmax(axis=0), np.arange(overlaps.shape[1])]
    overlaps_groundtruth_max = np.where(overlaps == overlaps_groundtruth_max)[0]

    labels[overlaps_max < rpn_negative_overlap] = 0
    labels[overlaps_groundtruth_max] = 1
    labels[overlaps_max >= rpn_positive_overlap] = 1

    # Reduce the number if necessary
    cnt_foreground = int(rpn_fg_fraction * RPN_BATCH_SIZE)
    foreground_indices = np.where(labels == 0)[0]
    if len(foreground_indices) > cnt_foreground:
        disabled_indices = np.random.choice(foreground_indices, size=len(foreground_indices) - cnt_foreground, replace=False)
        labels[disabled_indices] = -1

    cnt_background = RPN_BATCH_SIZE - np.sum(labels == 1)
    background_indices = np.where(labels == 0)[0]
    if len(background_indices) > cnt_background:
        disabled_indices = np.random.choice(background_indices, size=len(background_indices) - cnt_background, replace=False)
        labels[disabled_indices] = -1


def main():
    image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, None, 3])
    im_info = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 3])
    ground_truth = tf.placeholder(tf.float32, shape=[None, 5])

    inputs_layer = tf.placeholder(tf.float32, shape=[BATCH_SIZE, None, None, 3])

    conv_layers = vgg16_conv(inputs_layer)

    rpn_cls_prob, rpn_bbox_pred = rpn_layers(conv_layers)

    height = tf.to_int32(tf.ceil(im_info[0, 0] / FEAT_STRIDE))
    width = tf.to_int32(tf.ceil(im_info[0, 1] / FEAT_STRIDE))
    anchors = tf.py_func(generate_anchors,
                         [height, width, FEAT_STRIDE, ANCHOR_SCALES],
                         [tf.float32],
                         name='generate_anchors')
    roi, roi_score = tf.py_func(proposal_layer,
                                [rpn_cls_prob, rpn_bbox_pred, im_info, FEAT_STRIDE, anchors, ANCHOR_SCALES],
                                [tf.float32, tf.float32],
                                name='proposal')
    roi.set_shape([None, 5])
    roi_score.set_shape([None, 1])

    rpn_labels, \
    rpn_bbox_target, \
    rpn_bbox_inside_weights, \
    rpn_bbox_outside_weights = tf.py_func(anchor_target_layer,
                                          [rpn_cls_prob, ground_truth, im_info, FEAT_STRIDE, anchors, ANCHOR_SCALES],
                                          [tf.float32, tf.float32, tf.float32, tf.float32],
                                          name='anchor_target')


if __name__ == '__main__':
    main()
