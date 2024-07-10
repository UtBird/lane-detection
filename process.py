import cv2
import numpy as np
import line_function as lf 
import forward_display as fw
import matplotlib.pyplot as plt

def process(image):
    # Resize the image
    image = cv2.resize(image, (960, 540))
    fi = fw.forward_display(image)
    
    # Convert the image to HSV
    hsv_converter = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask to detect shadows
    lower_shadow = np.array([0, 0, 100], dtype=np.uint8)
    upper_shadow = np.array([150, 50, 50], dtype=np.uint8)
    shadow_mask = cv2.inRange(hsv_converter, lower_shadow, upper_shadow)

    # Mask the shadows
    frame_no_shadow = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(shadow_mask))
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame_no_shadow, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    canny_image = cv2.Canny(gray, 200, 250)

    # Apply GaussianBlur
    blurred_image = cv2.GaussianBlur(canny_image, (5, 5), 0)

    # Apply region of interest
    cropped_image = lf.region_selection(blurred_image)

    # Histogram analysis to detect lane lines
    histogram = np.sum(cropped_image[cropped_image.shape[0]//2:, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(cropped_image.shape[0] / nwindows)
    nonzero = cropped_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = cropped_image.shape[0] - (window + 1) * window_height
        win_y_high = cropped_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
    
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if len(leftx) > 0 and len(lefty) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
    else:
        left_fit = [0, 0, 0]
    
    if len(rightx) > 0 and len(righty) > 0:
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        right_fit = [0, 0, 0]
    
    ploty = np.linspace(0, cropped_image.shape[0]-1, cropped_image.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    result = np.copy(image)
    
    # Draw the polynomial lines onto the result image
    for i in range(len(ploty)):
        if 0 <= int(left_fitx[i]) < result.shape[1]:
            cv2.circle(result, (int(left_fitx[i]), int(ploty[i])), 2, (0, 255, 0), -1)
        if 0 <= int(right_fitx[i]) < result.shape[1]:
            cv2.circle(result, (int(right_fitx[i]), int(ploty[i])), 2, (0, 0, 255), -1)
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(cropped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Combine the result with the original image
    result = cv2.addWeighted(result, 1, color_warp, 0.3, 0)
    
    # Display the processed frame
    cv2.imshow('lines', result)
    cv2.imshow('cropped', cropped_image)
    cv2.imshow('fi', fi)

# Bu fonksiyon ile process fonksiyonunu çağırabilirsiniz.
