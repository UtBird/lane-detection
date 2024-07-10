import cv2
import numpy as np
import matplotlib.pyplot as plt

def forward_display(image):
    # perspektif alanı seçme
            tl = [410, 310]
            tr = [600, 320]
            br = [930, 530]
            bl = [90, 535]

            corner_points_array = np.float32([tl,tr,br,bl])

            # original image dimensions
            width = 1895
            height = 1052

            pts1 = np.float32([tl,bl,tr,br])
            pts2 = np.float32([[0,0],[0,480],[640,0],[640,480]])

            #warplama ve kuş bakışı alma 
            matrix =cv2.getPerspectiveTransform(pts1,pts2)
            tranformed_frame = cv2.warpPerspective(image,matrix,(640,480))
            return tranformed_frame
            
