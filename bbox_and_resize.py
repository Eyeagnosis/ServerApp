import cv2
import glob, os
import numpy as np

def process(NAME, DIM):
    img = cv2.imread(NAME)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    maxA = -1
    ind = -1
    for i in range(len(contours)):
        x = cv2.contourArea(contours[i])
        if x > maxA:
             maxA = x
             ind = i

    cnt = contours[ind]
    x,y,w,h = cv2.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    resized = cv2.resize(crop, DIM)
    cv2.imwrite(NAME,resized)
if __name__ == "__main__":
    count = 0
    
    img_files = glob.glob("train/**/*.jpeg", recursive=True)
    img_files += glob.glob("validation/**/*.jpeg", recursive=True)
    img_files += glob.glob("test/**/*.jpeg", recursive=True)
    img_files += glob.glob("unused/**/*.jpeg", recursive=True)
    for i in img_files:
        basename = os.path.basename(i)
        fn, _ = os.path.splitext(basename)
        os.rename(i, "train/{0}".format(basename))
    
    for infile in glob.glob("train/*.jpeg"):
        process(infile, (768, 768))
        count += 1
        if count % 5000 == 0:
            print(count)
