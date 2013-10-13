import numpy as np
import cv2
import os
from operator import itemgetter


class champ:
    def __init__(self,fileName,imgPath):
        self.name = fileName.split('.')[0]  #Split by dot in file name
        self.image = cv2.imread(imgPath)
        self.r_ch = self.image[:][:][0]  # Might be right indexing. lol
        self.g_ch = self.image[:][:][1]
        self.b_ch = self.image[:][:][2]

        self.r_hist = cv2.calcHist([self.image],[0],None,[256],[0,255])
        self.g_hist = cv2.calcHist([self.image],[1],None,[256],[0,255])
        self.g_hist = cv2.calcHist([self.image],[2],None,[256],[0,255])


imdir = './Champions/'
imf = os.listdir(imdir)
imf = imf[3:]

champList= []
for ind,champFile in enumerate(imf):
    newChamp = champ(champFile, ''.join([imdir,champFile]))
    champList.append(newChamp)

champSize = 120


base = cv2.imread('./img/vi.png')
BL = base[len(base)/4:len(base):1,0:len(base):1]

#Magic Box

# charBox= [14,688,92, 92]
charBox = [16,720,74,68]
xPos = charBox[0]
yPos = charBox[1]
sL = charBox[2]
scale = champSize/sL
# cv2.rectangle(BL,(14,688),(14+92,688+92),(0,0,255),5)
# cv2.imshow("box",BL)

cropped = BL[ yPos:yPos+sL,xPos:xPos+sL]
bigCrop = cv2.resize(cropped ,(champSize,champSize))
dist= []
bigCrop = np.array(bigCrop,dtype='f')
for ii,champ in enumerate(champList):
    dist.append(sum(sum(sum(abs( bigCrop -champ.image)))))

winner= min(enumerate(dist), key=itemgetter(1))[0]
print("Champion is: {0}").format(champList[winner].name)
cv2.imshow(champList[winner].name,champList[winner].image)
cv2.waitKey()











# ## FIND THE SQUARE
# # I = rgb2gray(BL);
# # th = graythresh(I);
# # I_th = im2bw(I,th);
# I = cv2.cvtColor(BL, cv2.COLOR_BGR2GRAY)
# (thresh, I_th) = cv2.threshold(I, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
# # Ifill = imfill(I_th,'holes');
# Ifill = I_th
# contour, hier = cv2.findContours(Ifill,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
# for cnt in contour:
#     cv2.drawContours(Ifill,[cnt], 0, 255, -1)
#
#  # Iarea = bwareaopen(Ifill,100);
# gray = Ifill
# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
# Iarea = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)
#


# edges = cv2.Canny(I,150,200,apertureSize = 3)
# lines = cv2.HoughLines(edges,1,np.pi/180,275)
# img =BL.copy()
# for rho,theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))   # Here i have used int() instead of rounding the decimal value, so 3.8 --> 3
#     y1 = int(y0 + 1000*(a))    # But if you want to round the number, then use np.around() function, then 3.8 --> 4.0
#     x2 = int(x0 - 1000*(-b))   # But we need integers, so use int() function after that, ie int(np.around(x))
#     y2 = int(y0 - 1000*(a))
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imshow('houghlines',img)
#


# lines = cv2.HoughLinesP(edges,1,np.pi/180,150, minLineLength = 100, maxLineGap = 10)
# for x1,y1,x2,y2 in lines[0]:
#     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
# cv2.imshow('houghlines',img)



# contour,hier = cv2.findContours(Iarea,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
# Ifinal = Iarea.copy()
# for cnt in contour:
#     cv2.drawContours(Ifinal,[cnt], 0, 255, -1)
#
# cv2.imshow("stuff",Ifinal)
# cv2.waitKey()
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