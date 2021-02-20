import numpy as np

import cv2

import torch

from scipy.spatial import ConvexHull


# Reference: https://bitbucket.org/william_rusnack/minimumboundingbox/src/master/
# This helps to generate the rotated rectangle with minimum area that covers the
# quadrangle bbox ground. It uses convex hull under the hoods to solve this problem.
# The entire concept is well explained here: https://stackoverflow.com/q/13542855/5353128
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an n * 2 matrix of coordinates
    :rval: an n * 2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = hull_points[1:] - hull_points[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T

    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def icdar_collate(batch):
    """
    Collate function for ICDAR dataset. It receives a batch of ground truths
    and formats it in required format.
    """
    image_paths, img, boxes, training_mask, transcripts, score_map, geo_map = zip(*batch)
    batch_size = len(score_map)
    images, score_maps, geo_maps, training_masks = [], [], [], [] 

    # convert all numpy arrays to tensors
    for idx in range(batch_size):
        if img[idx] is not None:
            images.append(torch.from_numpy(img[idx]).permute(2, 0, 1))
            score_maps.append(torch.from_numpy(score_map[idx]).permute(2, 0, 1))
            geo_maps.append(torch.from_numpy(geo_map[idx]).permute(2, 0, 1))
            training_masks.append(torch.from_numpy(training_mask[idx]).permute(2, 0, 1))

    images = torch.stack(images, 0)
    score_maps = torch.stack(score_maps, 0)
    geo_maps = torch.stack(geo_maps, 0)
    training_masks = torch.stack(training_masks, 0)

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

    return image_paths, images, bboxs, training_masks, texts, score_maps, geo_maps, mapping


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
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((img_h, img_w), dtype = np.uint8)

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

        # if the poly is too small, then ignore it during training
        bbox_h = min(np.linalg.norm(bbox[0] - bbox[3]), np.linalg.norm(bbox[1] - bbox[2]))
        bbox_w = min(np.linalg.norm(bbox[0] - bbox[1]), np.linalg.norm(bbox[2] - bbox[3]))
        
        if min(bbox_h, bbox_w) < 10:
            cv2.fillPoly(training_mask, bbox.astype(np.int32)[np.newaxis, :, :], 0)
        # if tag:
        #     cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)

        # Get all the points (in current bbox) in a helper mask
        bbox_points = np.argwhere(bbox_mask == (bbox_idx+1))

        # Now, as per the assumption, the bbox can be of any shape (quadrangle).
        # Therefore, to get the angle of rotation and pixel distances from the
        # bbox edges, fit a minimum area rectangle to bbox quadrangle.
        rectangle = minimum_bounding_rectangle(bbox)
        rectangle, rotation_angle = _align_vertices(rectangle)
        final_bboxes.append(rectangle)  # TODO: Filter very small bboxes here

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
    training_mask = training_mask[::4, ::4, np.newaxis].astype(np.float32)

    return score_map, geo_map, training_mask, np.vstack(final_bboxes)


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
