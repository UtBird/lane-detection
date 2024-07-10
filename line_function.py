import cv2
import numpy as np

def region_selection(image):
    height, width = image.shape[:2]
    mask = np.zeros_like(image)
    
    # Define the region of interest vertices
    polygons = np.array([[
        (int(0.1 * width), height),
        (int(0.9 * width), height),
        (int(0.55 * width), int(0.6 * height)),
        (int(0.45 * width), int(0.6 * height))
    ]], dtype=np.int32)
    
    # Fill the region of interest with white color
    cv2.fillPoly(mask, polygons, 255)
    
    # Perform a bitwise AND to get the ROI
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_lines(image, lines, color=[0, 0, 255], thickness=2):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined_image

def hough_transform(canny_image):
    lines = cv2.HoughLinesP(canny_image, 
                            rho=2, 
                            theta=np.pi/180, 
                            threshold=100, 
                            lines=np.array([]), 
                            minLineLength=40, 
                            maxLineGap=5)
    return lines

def process_frame(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 50, 150)
    
    cropped_image = region_selection(canny_image)
    lines = hough_transform(cropped_image)
    
    line_image = draw_lines(image, lines)
    return line_image
