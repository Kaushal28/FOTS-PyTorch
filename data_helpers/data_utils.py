import numpy as np

import cv2

import torch

from shapely.geometry import Polygon

def icdar_collate(batch):
    """
    Collate function for ICDAR dataset. It receives a batch of ground truths
    and formats it in required format.
    """
    image_paths, img, boxes, transcripts, score_map, geo_map = zip(*batch)
    batch_size = len(score_map)
    images, score_maps, geo_maps = [], [], []

    # convert all numpy arrays to tensors
    for idx in range(batch_size):
        if img[idx] is not None:
            images.append(torch.from_numpy(img[idx]).permute(2, 0, 1))
            score_maps.append(torch.from_numpy(score_map[idx]).permute(2, 0, 1))
            geo_maps.append(torch.from_numpy(geo_map[idx]).permute(2, 0, 1))

    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)

    texts, bboxs, mapping = [], [], []
    for idx, (text, bbox) in enumerate(zip(transcripts, boxes)):
        for txt, box in zip(text, bbox):
            mapping.append(idx)
            texts.append(txt)
            bboxs.append(box)

    mapping = np.array(mapping)
    texts = np.array(texts)
    bboxs = np.stack(bboxs, axis=0)
    bboxs = np.concatenate([bboxs, np.ones((len(bboxs), 1))], axis = 1).astype(np.float32)

    return image_paths, images, bboxs, texts, score_maps, geo_maps, mapping


def l2_norm(p1, p2=np.array([0, 0])):
    """
    Calculates the L2 norm (euclidean distance) between given two points.

    point (pi) should be in format (x, y)
    """
    return np.linalg.norm(p1 - p2)


def shrink_bbox(bbox, reference_lens, shrink_ratio=0.3):
    """
    Shrink the given bbox by given ratio.
    
    It first shrinks the two longer edges of a quadrangle, and then the
    two shorter ones. For each pair of two opposing edges, it determines
    the “longer” pair by comparing the mean of their lengths.

    For each edge (pi, p(i mod 4)+1),
    it shrinks it by moving its two endpoints inward along the
    edge by shrink_ratio*reference_lens[i] and 
    shrink_ratio*reference_lens[(i mod 4)+1] respectively.

    bbox shape: (4, 2) (clock wise from top left).
    """

    reference_lens = shrink_ratio * reference_lens

    # First find the "longer" edge pair
    if (
        # top horizontal edge + bottom horizontal edge
        l2_norm(bbox[0] - bbox[1]) + l2_norm(bbox[2] - bbox[3]) >
        # left vertical edge + right vertical edge
        l2_norm(bbox[0] - bbox[3]) + l2_norm(bbox[1] - bbox[2])
    ):
        # This means pair of horizontal edge is "longer" than pair
        # of vertical edges. So first shrink (p0, p1) and (p2, p3)
        # then shrink (p1, p2) and (p3, p0)

        # angle of edge between p0 and p1. Which is tan-1((y2-y1)/(x2-x1))
        theta = np.arctan2((bbox[1][1] - bbox[0][1]), (bbox[1][0] - bbox[0][0]))
        bbox[0][0] += reference_lens[0] * np.cos(theta)
        bbox[0][1] += reference_lens[0] * np.sin(theta)
        bbox[1][0] -= reference_lens[1] * np.cos(theta)
        bbox[1][1] -= reference_lens[1] * np.sin(theta)

        # shrink p2 and p3
        theta = np.arctan2((bbox[2][1] - bbox[3][1]), (bbox[2][0] - bbox[3][0]))
        bbox[2][0] -= reference_lens[2] * np.cos(theta)
        bbox[2][1] -= reference_lens[2] * np.sin(theta)
        bbox[3][0] += reference_lens[3] * np.cos(theta)
        bbox[3][1] += reference_lens[3] * np.sin(theta)

        # Then shrink p0 and p3
        theta = np.arctan2((bbox[3][0] - bbox[0][0]), (bbox[3][1] - bbox[0][1]))
        bbox[0][0] += reference_lens[0] * np.sin(theta)
        bbox[0][1] += reference_lens[0] * np.cos(theta)
        bbox[3][0] -= reference_lens[3] * np.sin(theta)
        bbox[3][1] -= reference_lens[3] * np.cos(theta)

        # shrink p1 and p2
        theta = np.arctan2((bbox[2][0] - bbox[1][0]), (bbox[2][1] - bbox[1][1]))
        bbox[1][0] += reference_lens[1] * np.sin(theta)
        bbox[1][1] += reference_lens[1] * np.cos(theta)
        bbox[2][0] -= reference_lens[2] * np.sin(theta)
        bbox[2][1] -= reference_lens[2] * np.cos(theta)
    else:
        # This means pair of vertical edge is "longer" than pair
        # of horizontal edges. So first shrink (p1, p2) and (p3, p0)
        # then shrink (p0, p1) and (p2, p3)
        theta = np.arctan2((bbox[3][0] - bbox[0][0]), (bbox[3][1] - bbox[0][1]))
        bbox[0][0] += reference_lens[0] * np.sin(theta)
        bbox[0][1] += reference_lens[0] * np.cos(theta)
        bbox[3][0] -= reference_lens[3] * np.sin(theta)
        bbox[3][1] -= reference_lens[3] * np.cos(theta)
        # shrink p1, p2
        theta = np.arctan2((bbox[2][0] - bbox[1][0]), (bbox[2][1] - bbox[1][1]))
        bbox[1][0] += reference_lens[1] * np.sin(theta)
        bbox[1][1] += reference_lens[1] * np.cos(theta)
        bbox[2][0] -= reference_lens[2] * np.sin(theta)
        bbox[2][1] -= reference_lens[2] * np.cos(theta)
        # shrink p0, p1
        theta = np.arctan2((bbox[1][1] - bbox[0][1]), (bbox[1][0] - bbox[0][0]))
        bbox[0][0] += reference_lens[0] * np.cos(theta)
        bbox[0][1] += reference_lens[0] * np.sin(theta)
        bbox[1][0] -= reference_lens[1] * np.cos(theta)
        bbox[1][1] -= reference_lens[1] * np.sin(theta)
        # shrink p2, p3
        theta = np.arctan2((bbox[2][1] - bbox[3][1]), (bbox[2][0] - bbox[3][0]))
        bbox[3][0] += reference_lens[3] * np.cos(theta)
        bbox[3][1] += reference_lens[3] * np.sin(theta)
        bbox[2][0] -= reference_lens[2] * np.cos(theta)
        bbox[2][1] -= reference_lens[2] * np.sin(theta)

    return bbox


def _point_to_line_dist(p1, p2, p3):
    """
    Find perpendicular distance from point p3 to line passing through
    p1 and p2.

    Reference: https://stackoverflow.com/a/39840218/5353128
    """
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def _bbox_angle(bbox):
    """
    Find the angle of rotation of given bbox.

    As mentioned in the EAST paper (https://arxiv.org/pdf/1704.03155.pdf),
    the bbox angle ground truth will the the angle between the horizontal
    line and bottom edge of the given bbox (rectangle).
    """
    # Get the bottom edge
    p2, p3 = bbox[2], bbox[3]
    # The angle of any line can be given by tan-1((y2-y1) / (x2-x1))
    return np.arctan2(p2[1] - p3[1], p2[0] - p3[0])


def fit_line(p1, p2):
    # fit a line ax+by+c = 0
    if p1[0] == p1[1]:
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg = 1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype = np.float32)


def _align_vertices(bbox):
    """
    Align (sort) the vertices of the given bbox (rectangle) in such a way
    that the base of the rectangle forms minimum angle with horizontal axis.
    This is required because a single rectangle can be written in many
    ways (just by rotating the vertices in the list notation) such that the
    base of the rectangle will get changed in different notations and will form
    the angle which is multiple of original minimum angle.

    Reference: EAST implementation for ICDAR-2015 dataset:
    https://github.com/argman/EAST/blob/dca414de39a3a4915a019c9a02c1832a31cdd0ca/icdar.py#L352
    """
    p_lowest = np.argmax(bbox[:, 1])
    if np.count_nonzero(bbox[:, 1] == bbox[p_lowest, 1]) == 2:
        # This means there are two points in the horizantal axis (because two lowest points).
        # That means 0 angle.
        # The bottom edge is parallel to the X-axis, then p0 is the upper left corner.
        p0_index = np.argmin(np.sum(bbox, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return bbox[[p0_index, p1_index, p2_index, p3_index]], 0.0
    else:
        # Find the point to the right of the lowest point.
        p_lowest_right = (p_lowest - 1) % 4
        angle = np.arctan(
            -(bbox[p_lowest][1] - bbox[p_lowest_right][1]) / (bbox[p_lowest][0] - bbox[p_lowest_right][0])
        )
        if angle / np.pi * 180 > 45:
            # Lowest point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            return bbox[[p0_index, p1_index, p2_index, p3_index]], -(np.pi/2 - angle)
        else:
            # Lowest point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            return bbox[[p0_index, p1_index, p2_index, p3_index]], angle


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    '''
    fit a rectangle from a parallelogram
    :param poly:
    :return:
    '''
    p0, p1, p2, p3 = poly
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    if angle_p0 < 0.5 * np.pi:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p0 and p2
            ## p0
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)

            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            ## p2
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)

            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype = np.float32)
        else:
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)

            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)

            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype = np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # p1 and p3
            ## p1
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)

            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            ## p3
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)

            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype = np.float32)
        else:
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)

            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)

            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype = np.float32)


def generate_rbbox(image, bboxes, transcripts):
    """
    Generate RBOX (Rotated bbox) as per this paper:
    https://arxiv.org/pdf/1704.03155.pdf
    """
    img_h, img_w, _ = image.shape
    # geo_map is pixel/bbox location map which stores distances of
    # pixels from top, right, bottom and left from corresponding bbox edges
    # and angle of rotation of the bbox (4+1=5 channels).
    geo_map = np.zeros((img_h, img_w, 5), dtype = np.float32)
    # Single channel which indicates whether the pixel is part of text or
    # background.
    score_map = np.zeros((img_h, img_w), dtype = np.uint8)
    # Temporary bbox mask which is used as a helper.
    bbox_mask = np.zeros((img_h, img_w), dtype = np.uint8)

    # For each bbox, first shrink the bbox as per the paper.
    # For that, for each bbox, calculate the reference length (r_i)
    # for each bbox vertex p_i.
    
    final_bboxes = []
    for bbox_idx, bbox in enumerate(bboxes):
        # Reference length calculation
        reference_lens = []
        for idx in range(1, 5):
            reference_lens.append(
                min(
                    l2_norm(bbox[idx-1], bbox[(idx%4)+1-1]),
                    l2_norm(bbox[idx-1], bbox[((idx+2)%4)+1-1]),
                )
            )
        
        shrink_ratio = 0.3  # from papar
        shrunk_bbox = shrink_bbox(bbox.copy(), np.array(reference_lens), shrink_ratio).astype(np.int32)[np.newaxis, :, :]

        cv2.fillPoly(score_map, shrunk_bbox, 1)
        cv2.fillPoly(bbox_mask, shrunk_bbox, bbox_idx+1)

        # Get all the points (in current bbox) in a helper mask
        bbox_points = np.argwhere(bbox_mask == (bbox_idx+1))

        # Now, as per the assumption, the bbox can be of any shape (quadrangle).
        # Therefore, to get the angle of rotation and pixel distances from the
        # bbox edges, fit a minimum area rectangle to bbox quadrangle.
        fitted_parallelograms = []
        for i in range(4):
            p0 = bbox[i]
            p1 = bbox[(i + 1) % 4]
            p2 = bbox[(i + 2) % 4]
            p3 = bbox[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if _point_to_line_dist(p0, p1, p2) > _point_to_line_dist(p0, p1, p3):
                # 平行线经过p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # 经过p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if _point_to_line_dist(p1, new_p2, p0) > _point_to_line_dist(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p1 = p1
            new_p2 = p2
            new_p3 = p3
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if _point_to_line_dist(p0, p3, p1) > _point_to_line_dist(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        areas = [Polygon(t).area for t in fitted_parallelograms]
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype = np.float32)
        # sort thie polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis = 1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]

        rectangle = rectangle_from_parallelogram(parallelogram)
        rectangle, rotation_angle = _align_vertices(rectangle)
        final_bboxes.append(rectangle.flatten())  # TODO: Filter very small bboxes here

        # This rectangle has 4 vertices as required. Now, we can construct
        # the geo_map.
        for bbox_y, bbox_x in bbox_points:
            bbox_point = np.array([bbox_x, bbox_y], dtype=np.float32)
            # distance from top
            geo_map[bbox_y, bbox_x, 0] = _point_to_line_dist(rectangle[0], rectangle[1], bbox_point)
            # distance from right
            geo_map[bbox_y, bbox_x, 1] = _point_to_line_dist(rectangle[1], rectangle[2], bbox_point)
            # distance from bottom
            geo_map[bbox_y, bbox_x, 2] = _point_to_line_dist(rectangle[2], rectangle[3], bbox_point)
            # distance from left
            geo_map[bbox_y, bbox_x, 3] = _point_to_line_dist(rectangle[3], rectangle[0], bbox_point)
            # bbox rotation angle
            geo_map[bbox_y, bbox_x, 4] = rotation_angle

    # Size of the feature map from shared convolutions is 1/4th of
    # original image size. So all this geo_map should be of the
    # same size.
    score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)
    geo_map = geo_map[::4, ::4].astype(np.float32)

    return score_map, geo_map, np.vstack(final_bboxes)


def resize_image(image, image_size):
    """
    Resize the given image to image_size * image_size
    shaped square image.
    """
    # First pad the given image to match the image_size or image's larger
    # side (whichever is larger). [Create a square image]
    img_h, img_w, _ = image.shape
    max_size = max(image_size, img_w, img_h)

    # Create new square image of appropriate size
    img_padded = np.zeros((max_size, max_size, 3), dtype=np.float32)
    # Copy the original image into new image
    # (basically, new image is padded version of original image).
    img_padded[:img_h, :img_w, :] = image.copy()
    img_h, img_w, _ = img_padded.shape

    # if image_size higher that image sides, then the current padded
    # image will be of size image_size * image_size. But if not, resize the
    # padded iamge. This is done to keep the aspect ratio same even after
    # square resize.
    img_padded = cv2.resize(img_padded, dsize=(image_size, image_size))

    # We need the ratio of resized image width and heights to its
    # older dimensions to scale the bounding boxes accordingly
    scale_x = image_size / img_w
    scale_y = image_size / img_h

    return img_padded, scale_x, scale_y