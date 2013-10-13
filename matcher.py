import numpy as np
import cv2
import os
from operator import itemgetter


class matcher:
    class champ:
        def __init__(self,fileName,imgPath):
            self.name = fileName.split('.')[0]  # Split by dot in file name
            self.image = cv2.imread(imgPath)
            self.r_ch = self.image[:][:][0]  # Might be right indexing. lol
            self.g_ch = self.image[:][:][1]
            self.b_ch = self.image[:][:][2]

            self.r_hist = cv2.calcHist([self.image],[0],None,[256],[0,255])
            self.g_hist = cv2.calcHist([self.image],[1],None,[256],[0,255])
            self.g_hist = cv2.calcHist([self.image],[2],None,[256],[0,255])
    def matchChamp(self,filePath):
        imdir = './Champions/'
        imf = os.listdir(imdir)
        imf = imf[3:]

        champList= []
        for ind,champFile in enumerate(imf):
            newChamp = self.champ(champFile, ''.join([imdir,champFile]))
            champList.append(newChamp)

        champSize = champList[0].image.shape[0] #Should be 120

        base = cv2.imread(filePath)
        BL = base[len(base)/2:len(base):1,0:len(base)/4:1]


        ## FIND THE SQUARE
        # Creating a clean Binary Image
        I = cv2.cvtColor(BL, cv2.COLOR_BGR2GRAY)
        (thresh, I_th) = cv2.threshold(I, 128, 255,  cv2.THRESH_OTSU )


        # cv2.imshow("ifill",I_th)

        Ifill = I_th.copy()
        contour, _ = cv2.findContours(Ifill,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contour:
            cv2.drawContours(Ifill,[cnt], 0, 255, -1)

        gray = I_th
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        Iarea = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
        # Use Binary Image to get bounding boxes with squareness metrics

        sq_thresh = 0.1
        sq1 = []
        sq2 =[]
        #contours, _ = cv2.findContours(Iarea,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours1, _ = cv2.findContours(I_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(Iarea,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours1:
            m = cv2.moments(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            sq_size = w*h;
            # area = m['m00']
            # cmpArea = area/sq_size
            sq_sqness = abs(1-max(w,h)/min(w,h))
            loc_bias = 1.0*x/BL.shape[1] + 1.0*(BL.shape[0]-y)/BL.shape[0]
            cv2.rectangle(BL,(x,y),(x+w,y+h),(0,0,255),2)
            sq1.append(sq_size * (sq_sqness<sq_thresh)/loc_bias)

        for cnt in contours2:
            m = cv2.moments(cnt)
            x,y,w,h = cv2.boundingRect(cnt)
            sq_size = w*h;
            # area = m['m00']
            # cmpArea = area/sq_size
            sq_sqness = abs(1-max(w,h)/min(w,h))
            loc_bias = 1.0*x/BL.shape[1] + 1.0*(BL.shape[0]-y)/BL.shape[0]
            cv2.rectangle(BL,(x,y),(x+w,y+h),(0,0,255),2)
            sq2.append(sq_size * (sq_sqness<sq_thresh)/loc_bias)

        # (85<w<100)*(85<h<100)
        ind1= max(enumerate(sq1), key=itemgetter(1))[0]
        ind2= max(enumerate(sq2), key=itemgetter(1))[0]

        ind = max(enumerate([sq1[ind1], sq2[ind2]]), key=itemgetter(1))[0]

        if(ind == 0):
            x,y,w,h = cv2.boundingRect(contours1[ind1])
        elif(ind ==1):
            x,y,w,h = cv2.boundingRect(contours2[ind2])
        #x,y,w,h = cv2.boundingRect(contours[ind])

        # cv2.rectangle(BL,(x,y),(x+w,y+h),(0,255,0),5)
        # print(w)
        # print(h)
        sl = min(w,h); # Use as side length
        cv2.rectangle(BL,(x,y),(x+sl,y+sl),(0,255,0),5)

        cropped = BL[ y:y+sl,x:x+sl]
        bigCrop = cv2.resize(cropped ,(champSize,champSize))
        dist= []
        bigCrop = np.array(bigCrop,dtype='f')
        for ii,champ in enumerate(champList):
            dist.append(sum(sum(sum(abs( bigCrop -champ.image)))))

        winner= min(enumerate(dist), key=itemgetter(1))[0]
        cv2.imshow("Original",BL)
        cv2.imshow(champList[winner].name,champList[winner].image)
        cv2.waitKey()
        return champList[winner].name
