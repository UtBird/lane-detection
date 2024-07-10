import cv2
import numpy as np
import matplotlib.pyplot as plt

def pixel_points(y1, y2, line):
    """
    çiizgileribn kordinatlarını belirleme
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    """
    sağ sol şerit noktalar
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line

def average_slope_intercept(lines):
    """
    ortalama eğim hesaplama
    """
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            # calculating slope of a line
            slope = (y2 - y1) / (x2 - x1)
            # calculating intercept of a line
            intercept = y1 - (slope * x1)
            # calculating length of a line
            length = np.sqrt(((y2 - y1) ** 2) + ((x2 - x1) ** 2))
            # slope of left lane is negative and for the right lane slope is positive
            if slope < -0.5:  # sol şerit
                left_lines.append((slope, intercept))
                left_weights.append((length))
            elif slope > 0.5:  # sağ şerit
                right_lines.append((slope, intercept))
                right_weights.append((length))

    # sol ve sağ şeritlerin ortalama eğim ve intercept değerlerini bul
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    return left_lane, right_lane

def region_selection(image):
    """
    alan belirleme ve maskeleme
    """
    # create an array of the same size as the input image
    mask = np.zeros_like(image)
    # if you pass an image with more than one channel
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        # color of the mask polygon (white)
        ignore_mask_color = 255
    # creating a polygon to focus only on the road in the picture
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    # filling the polygon with white color and generating the final mask
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    # performing Bitwise AND on the input image and mask to get only the edges on the road
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(frame, lines, color=(0, 255, 0), thickness=5):
    """
    çizgi çizme 
    """
    line_image = np.zeros_like(frame)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    frame_with_lines = cv2.addWeighted(frame, 0.8, line_image, 1, 0)
    return frame_with_lines