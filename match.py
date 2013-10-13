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


base = cv2.imread('./img/nid_dead.png')
BL = base[len(base)/4:len(base):1,0:len(base)/4:1]

#Magic Box
#
# # charBox= [14,688,92, 92]
# charBox = [16,720,74,68]
# xPos = charBox[0]
# yPos = charBox[1]
# sL = charBox[2]
# scale = champSize/sL
# # cv2.rectangle(BL,(14,688),(14+92,688+92),(0,0,255),5)
# # cv2.imshow("box",BL)
#
#


## FIND THE SQUARE
# I = rgb2gray(BL);
# th = graythresh(I);
# I_th = im2bw(I,th);
I = cv2.cvtColor(BL, cv2.COLOR_BGR2GRAY)
(thresh, I_th) = cv2.threshold(I, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# Ifill = imfill(I_th,'holes');
Ifill = I_th
contour, _ = cv2.findContours(Ifill,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contour:
    cv2.drawContours(Ifill,[cnt], 0, 255, -1)

# Iarea = bwareaopen(Ifill,100);
gray = Ifill
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
Iarea = cv2.morphologyEx(gray,cv2.MORPH_OPEN,kernel)

sq_thresh = 0.2
sq = []
contours, _ = cv2.findContours(Iarea,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    sq_size = w*h;
    sq_sqness = abs(1-max(w,h)/min(w,h))
    loc_bias = 1.0*x/BL.shape[1] + 1.0*(BL.shape[0]-y)/BL.shape[0]
    sq.append(sq_size * (sq_sqness<sq_thresh)/loc_bias)


ind= max(enumerate(sq), key=itemgetter(1))[0]
x,y,w,h = cv2.boundingRect(contours[ind])
cv2.rectangle(BL,(x,y),(x+w,y+h),(0,0,255),5)
cv2.imshow("Orig",BL)







cropped = BL[ y:y+h,x:x+w]
bigCrop = cv2.resize(cropped ,(champSize,champSize))
dist= []
bigCrop = np.array(bigCrop,dtype='f')
for ii,champ in enumerate(champList):
    dist.append(sum(sum(sum(abs( bigCrop -champ.image)))))

winner= min(enumerate(dist), key=itemgetter(1))[0]
print("Champion is: {0}").format(champList[winner].name)
cv2.imshow(champList[winner].name,champList[winner].image)

cv2.waitKey()

# BW = Iarea
# # grab contours
# cs,_ = cv2.findContours( BW.astype('uint8'), mode=cv2.RETR_LIST,
#                              method=cv2.CHAIN_APPROX_SIMPLE )
# set up the 'FilledImage' bit of regionprops.
# filledI = np.zeros(BW.shape[0:2]).astype('uint8')
# # set up the 'ConvexImage' bit of regionprops.
# convexI = np.zeros(BW.shape[0:2]).astype('uint8')
#
# # for each contour c in cs:
# # will demonstrate with cs[0] but you could use a loop.
# i=0
# c = cs[i]
#
# # calculate some things useful later:
# m = cv2.moments(c)
#
# # ** regionprops **
# Area          = m['m00']
# Perimeter     = cv2.arcLength(c,True)
# # bounding box: x,y,width,height
# BoundingBox   = cv2.boundingRect(c)
# # centroid    = m10/m00, m01/m00 (x,y)
# Centroid      = ( m['m10']/m['m00'],m['m01']/m['m00'] )
#
# # EquivDiameter: diameter of circle with same area as region
# EquivDiameter = np.sqrt(4*Area/np.pi)
# # Extent: ratio of area of region to area of bounding box
# Extent        = Area/(BoundingBox[2]*BoundingBox[3])
#
# # FilledImage: draw the region on in white
# cv2.drawContours( filledI, cs, i, color=255, thickness=-1 )
# # calculate indices of that region..
# regionMask    = (filledI==255)
# # FilledArea: number of pixels filled in FilledImage
# FilledArea    = np.sum(regionMask)
# # PixelIdxList : indices of region.
# # (np.array of xvals, np.array of yvals)
# PixelIdxList  = regionMask.nonzero()
#
# # CONVEX HULL stuff
# # convex hull vertices
# ConvexHull    = cv2.convexHull(c)
# ConvexArea    = cv2.contourArea(ConvexHull)
# # Solidity := Area/ConvexArea
# Solidity      = Area/ConvexArea
# # convexImage -- draw on convexI
# cv2.drawContours( convexI, [ConvexHull], -1,
#                   color=255, thickness=-1 )
#
# # ELLIPSE - determine best-fitting ellipse.
# centre,axes,angle = cv2.fitEllipse(c)
# MAJ = np.argmax(axes) # this is MAJor axis, 1 or 0
# MIN = 1-MAJ # 0 or 1, minor axis
# # Note: axes length is 2*radius in that dimension
# MajorAxisLength = axes[MAJ]
# MinorAxisLength = axes[MIN]
# Eccentricity    = np.sqrt(1-(axes[MIN]/axes[MAJ])**2)
# Orientation     = angle
# EllipseCentre   = centre # x,y
#
# # ** if an image is supplied with the BW:
# # Max/Min Intensity (only meaningful for a one-channel img..)
# # MaxIntensity  = np.max(img[regionMask])
# # MinIntensity  = np.min(img[regionMask])
# # # Mean Intensity
# # MeanIntensity = np.mean(img[regionMask],axis=0)
# # pixel values
# PixelValues   = BL[regionMask]
# cv2.imshow("area",PixelValues)



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