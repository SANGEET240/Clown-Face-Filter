import cv2
import FaceMeshModule as fmm
from math import hypot
import numpy as np

# Object creation of class
detector = fmm.FaceMeshDetector(maxFaces = 1)

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    img, FaceLandmarks = detector.findFaceMesh(frame, draw = False)
    img, _ = detector.calculateFPS(img) 
    
    if FaceLandmarks:
        # -------------------------------Nose Section-------------------------------------
        
        
        
        FaceLandmarks = FaceLandmarks[0] # Extracting the list inside the list 'FaceLandMarks'
        # Extracting the X, Y point of the landmark Number 4, 48, 278
        Nose_Middle_PointX, Nose_Middle_PointY = FaceLandmarks[4][1], FaceLandmarks[4][2]
        Nose_Left_PointX, Nose_Left_PointY = FaceLandmarks[48][1], FaceLandmarks[48][2]
        Nose_Right_PointX, Nose_Right_PointY = FaceLandmarks[278][1], FaceLandmarks[278][2]
        Nose_Width = int(hypot(Nose_Left_PointX - Nose_Right_PointX, Nose_Left_PointY - Nose_Right_PointY))
        
        # Nose_IMG = cv2.resize(Nose_IMG, (Nose_Width, Nose_Width))
        cv2.circle(img, (Nose_Middle_PointX, Nose_Middle_PointY), (Nose_Width // 2) - 3, (0, 0, 209), -1)
        
        # -------------------------------Mouth circle section-------------------------------
        Left_MouthX, Left_MouthY = FaceLandmarks[57][1], FaceLandmarks[57][2]
        Right_MouthX, Right_MouthY = FaceLandmarks[287][1], FaceLandmarks[287][2] 
        Mouth_Width = int(hypot(Left_MouthX - Right_MouthX, Left_MouthY - Right_MouthY))
        
        cv2.circle(img, (Left_MouthX, Left_MouthY), (Mouth_Width // 8), (0, 0, 209), -1)
        cv2.circle(img, (Right_MouthX, Right_MouthY), (Mouth_Width // 8), (0, 0, 209), -1)
        
        
        
        # -------------------------------Eye Triangle Section--------------------------------
        
        
        
        # Left eye lower section-----------------
        # LELP_no_x/y = Left eye lower part [No = landmark number]
        LELP_1_x, LELP_1_y = FaceLandmarks[163][1], FaceLandmarks[163][2]
        LELP_2_x, LELP_2_y = FaceLandmarks[144][1], FaceLandmarks[144][2]
        LELP_3_x, LELP_3_y = FaceLandmarks[145][1], FaceLandmarks[145][2]
        LELP_4_x, LELP_4_y = FaceLandmarks[153][1], FaceLandmarks[153][2]
        # LELPMBP = Left eye lower part middle below point
        LELPMBPx, LELPMBPy = FaceLandmarks[119][1], FaceLandmarks[119][2] 
        
        pts = np.array([[LELP_1_x, LELP_1_y + 3], [LELP_2_x, LELP_2_y + 3], [LELP_3_x, LELP_3_y + 3], 
                        [LELP_4_x, LELP_4_y + 3], [LELPMBPx, LELPMBPy + 3]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img, [pts], color=(255, 151, 48))
        
        # Left eye upper section----------------
        # LEUP_no_x/y = Left eye upper part [No = landmark number]
        LEUP_1_x, LEUP_1_y = FaceLandmarks[161][1], FaceLandmarks[161][2]
        LEUP_2_x, LEUP_2_y = FaceLandmarks[160][1], FaceLandmarks[160][2]
        LEUP_3_x, LEUP_3_y = FaceLandmarks[159][1], FaceLandmarks[159][2]
        LEUP_4_x, LEUP_4_y = FaceLandmarks[158][1], FaceLandmarks[158][2]
        # LEUPMUP = Left eye lower part middle upper point
        LEUPMUPx, LEUPMUPy = FaceLandmarks[223][1], FaceLandmarks[223][2] 
        
        pts = np.array([[LEUP_1_x, LEUP_1_y - 3], [LEUP_2_x, LEUP_2_y - 3], [LEUP_3_x, LEUP_3_y - 3], 
                        [LEUP_4_x + 3, LEUP_4_y - 3], [LEUPMUPx + 4, LEUPMUPy - 3]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img, [pts], color=(255, 151, 48))
        
        # Right eye lower section-----------------
        # RELP_no_x/y = Right eye lower part [No = landmark number]
        RELP_1_x, RELP_1_y = FaceLandmarks[380][1], FaceLandmarks[380][2]
        RELP_2_x, RELP_2_y = FaceLandmarks[374][1], FaceLandmarks[374][2]
        RELP_3_x, RELP_3_y = FaceLandmarks[373][1], FaceLandmarks[373][2]
        RELP_4_x, RELP_4_y = FaceLandmarks[390][1], FaceLandmarks[390][2]
        # RELPMBP = Left eye lower part middle below point
        RELPMBPx, RELPMBPy = FaceLandmarks[348][1], FaceLandmarks[348][2]
        
        pts = np.array([[RELP_1_x, RELP_1_y + 3], [RELP_2_x, RELP_2_y + 3], [RELP_3_x, RELP_3_y + 3], 
                        [RELP_4_x, RELP_4_y + 3], [RELPMBPx, RELPMBPy + 3]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img, [pts], color=(255, 151, 48))
        
        # Right eye upper section----------------
        # REUP_no_x/y = Left eye upper part [No = landmark number]
        REUP_1_x, REUP_1_y = FaceLandmarks[384][1], FaceLandmarks[384][2]
        REUP_2_x, REUP_2_y = FaceLandmarks[385][1], FaceLandmarks[385][2]
        REUP_3_x, REUP_3_y = FaceLandmarks[386][1], FaceLandmarks[386][2]
        REUP_4_x, REUP_4_y = FaceLandmarks[387][1], FaceLandmarks[387][2] 
        # REUPMUP = Left eye lower part middle upper point
        REUPMUPx, REUPMUPy = FaceLandmarks[442][1], FaceLandmarks[442][2] 
        
        pts = np.array([[REUP_1_x, REUP_1_y - 3], [REUP_2_x, REUP_2_y - 3], [REUP_3_x, REUP_3_y - 3], 
                        [REUP_4_x + 3, REUP_4_y - 3], [REUPMUPx + 5, REUPMUPy - 3]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img, [pts], color=(255, 151, 48))
        
        
        
        # ---------------------------------LIPS Section------------------------------------
        
        
        
        # Lower lip section----------------------
        # LLUP_no_x/y = Lower Lip Upper Part [No = landmark number]
        LLUP_1_x, LLUP_1_y = FaceLandmarks[88][1], FaceLandmarks[88][2]
        LLUP_2_x, LLUP_2_y = FaceLandmarks[178][1], FaceLandmarks[178][2]
        LLUP_3_x, LLUP_3_y = FaceLandmarks[87][1], FaceLandmarks[87][2]
        LLUP_4_x, LLUP_4_y = FaceLandmarks[14][1], FaceLandmarks[14][2]
        LLUP_5_x, LLUP_5_y = FaceLandmarks[317][1], FaceLandmarks[317][2]
        LLUP_6_x, LLUP_6_y = FaceLandmarks[402][1], FaceLandmarks[402][2]
        LLUP_7_x, LLUP_7_y = FaceLandmarks[318][1], FaceLandmarks[318][2]
        
        # LLLP_no_x/y = Lower Lip Lower Part [No = landmark number]
        LLLP_1_x, LLLP_1_y = FaceLandmarks[181][1], FaceLandmarks[181][2]
        LLLP_2_x, LLLP_2_y = FaceLandmarks[84][1], FaceLandmarks[84][2]
        LLLP_3_x, LLLP_3_y = FaceLandmarks[17][1], FaceLandmarks[17][2]
        LLLP_4_x, LLLP_4_y = FaceLandmarks[314][1], FaceLandmarks[314][2]
        LLLP_5_x, LLLP_5_y = FaceLandmarks[405][1], FaceLandmarks[405][2]
        
        pts = np.array([[Left_MouthX, Left_MouthY], [LLUP_1_x, LLUP_1_y],
                        [LLUP_2_x, LLUP_2_y], [LLUP_3_x, LLUP_3_y], [LLUP_4_x, LLUP_4_y],
                        [LLUP_5_x, LLUP_5_y], [LLUP_6_x, LLUP_6_y], [LLUP_7_x, LLUP_7_y],
                        [Right_MouthX, Right_MouthY], 
                        
                        [LLLP_5_x, LLLP_5_y], [LLLP_4_x, LLLP_4_y], [LLLP_3_x, LLLP_3_y], 
                        [LLLP_2_x, LLLP_2_y], [LLLP_1_x, LLLP_1_y]],
                       np.int32)
        
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img, [pts], color=(0, 0, 209))
        
        # Upper lip section----------------------
        # ULUP_no_x/y = Upper Lip Upper Part [No = landmark number]
        ULUP_1_x, ULUP_1_y = FaceLandmarks[40][1], FaceLandmarks[40][2]
        ULUP_2_x, ULUP_2_y = FaceLandmarks[39][1], FaceLandmarks[39][2]
        ULUP_3_x, ULUP_3_y = FaceLandmarks[37][1], FaceLandmarks[37][2]
        ULUP_4_x, ULUP_4_y = FaceLandmarks[0][1], FaceLandmarks[0][2]
        ULUP_5_x, ULUP_5_y = FaceLandmarks[267][1], FaceLandmarks[267][2]
        ULUP_6_x, ULUP_6_y = FaceLandmarks[269][1], FaceLandmarks[269][2]
        ULUP_7_x, ULUP_7_y = FaceLandmarks[270][1], FaceLandmarks[270][2]
        
        # ULLP_no_x/y = Upper Lip Lower Part [No = landmark number]
        ULLP_1_x, ULLP_1_y = FaceLandmarks[80][1], FaceLandmarks[80][2]
        ULLP_2_x, ULLP_2_y = FaceLandmarks[81][1], FaceLandmarks[81][2]
        ULLP_3_x, ULLP_3_y = FaceLandmarks[82][1], FaceLandmarks[82][2]
        ULLP_4_x, ULLP_4_y = FaceLandmarks[13][1], FaceLandmarks[13][2]
        ULLP_5_x, ULLP_5_y = FaceLandmarks[312][1], FaceLandmarks[312][2]
        ULLP_6_x, ULLP_6_y = FaceLandmarks[311][1], FaceLandmarks[311][2]
        ULLP_7_x, ULLP_7_y = FaceLandmarks[310][1], FaceLandmarks[310][2]
        
        
        pts = np.array([[Left_MouthX, Left_MouthY], [ULUP_1_x, ULUP_1_y],
                        [ULUP_2_x, ULUP_2_y], [ULUP_3_x, ULUP_3_y], [ULUP_4_x, ULUP_4_y], 
                        [ULUP_5_x, ULUP_5_y], [ULUP_6_x, ULUP_6_y], [ULUP_7_x, ULUP_7_y],
                        [Right_MouthX, Right_MouthY],
                        
                        [ULLP_7_x, ULLP_7_y], [ULLP_6_x, ULLP_6_y], [ULLP_5_x, ULLP_5_y],
                        [ULLP_4_x, ULLP_4_y], [ULLP_3_x, ULLP_3_y], [ULLP_2_x, ULLP_2_y], 
                        [ULLP_1_x, ULLP_1_y]], 
                       np.int32)
        
        pts = pts.reshape((-1,1,2))
        cv2.fillPoly(img, [pts], color=(0, 0, 209))
        
        
        # print(FaceLandmarks)
    
    cv2.imshow("Clown Filter", img)
    if cv2.waitKey(1) & 0xFF == 27: # Escape button
        break
    
cap.release()
cv2.destroyAllWindows()