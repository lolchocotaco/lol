import numpy as np
import cv2
import os


class champ:
    def __init__(self,fileName,imgPath):
        self.name = fileName.split('.')[0]  #Split by dot in file name
        self.image = cv2.imread(imgPath)
        self.r_ch = self.image[:][:][0]  #Might be right indexing. lol
        self.g_ch = self.image[:][:][1]
        self.b_ch = self.image[:][:][2]

        self.r_hist = cv2.calcHist([self.image],[0],None,[256],[0,255])
        self.g_hist = cv2.calcHist([self.image],[1],None,[256],[0,255])
        self.g_hist = cv2.calcHist([self.image],[2],None,[256],[0,255])




base = cv2.imread('./img/thresh1.png')

BL = base[len(base)/4:len(base):1][0:len(base):1]
imdir = './Champions/'
imf = os.listdir(imdir)
imf = imf[3:]

champList= []
for ind,champFile in enumerate(imf):
    newChamp = champ(champFile, ''.join([imdir,champFile]))
    champList.append(newChamp)

## FIND THE SQUARE

I = cv2.cvtColor(BL, cv2.COLOR_BGR2GRAY)
(thresh, I_th) = cv2.threshold(I, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

des = cv2.bitwise_not(I_th)
contour,hier = cv2.findContours(des,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    cv2.drawContours(des,[cnt],0,255,-1)

gray = cv2.bitwise_not(des)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
res = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)

cv2.imshow("window",res)
#Ifill = imfill(I_th,'holes');
# Ifill = "something"
#
# #Iarea = bwareaopen(Ifill,100);
# Iarea = Ifill.copy()
# contours, hierarchy = cv2.findContours(Ifill,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
# for num,cnt in enumerate(contour):
#     area = cv2.contourArea(cnt)
#     if  area > 0 and area <= 100:
#          cv2.drawContours(Iarea, [cnt],0,255, -1);
#
# Ifinal,hier = cv2.findContours(IArea,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
#
#
# size = 200, 200, 3
# m = np.zeros(size, dtype=np.uint8)
# for num,cnt in enumerate(Ifinal):
#     print("hello")
# cv2.drawContours(mask,conour)












# Old copied code keeping for reference
# import cv2
#
# import numpy as np
# # from matplotlib import pyplot as plt
#
# img = cv2.imread('img/thresh1.png')
# img2 =np.copy(img)
# template = cv2.imread('img/truth/Thresh.png')
# w, h = template.shape[::-1]
#
# # All the 6 methods for comparison in a list
# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#             'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
#
# for i, meth in enumerate(methods):
#     img = img2.copy()
#     method = eval(meth)
#
#     # Apply template Matching
#     res = cv2.matchTemplate(img,template,method)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         top_left = min_loc
#     else:
#         top_left = max_loc
#     bottom_right = (top_left[0] + w, top_left[1] + h)
#
#     cv2.rectangle(img,top_left, bottom_right, 255, 2)
#
#     cv2.imwrite('res.png',img_rgb)
#
#     # plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     # plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     # plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     # plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     # plt.suptitle(meth)
#     #
#     # plt.show()