import os
import glob
from PIL import Image
import numpy as np
import cv2
dict=[
[  0,  0,  0 ],
[128, 64,128] ,
[244, 35,232],
[ 70, 70, 70],
[102,102,156],
[190,153,153],
[153,153,153],
[250,170, 30],
[220,220,  0],
[107,142, 35],
[152,251,152],
[ 70,130,180],
[220, 20, 60],
[255,  0,  0],
[  0,  0,142],
[  0,  0, 70],
[  0, 60,100],
[  0, 80,100],
[  0,  0,230],
[119, 11, 32]
]
path='/data2/gyang/TAGAN/results/test_340/images/real_A' + "/*"
photo_paths = glob.glob(path)
photo_paths = sorted(photo_paths)
for i, photo_path in enumerate(photo_paths):
    # print(photo_path)
    photo = cv2.imread(photo_path)
    cv2.cvtColor(photo, cv2.COLOR_BGR2RGB)
    # print(photo.shape[:2])
    gt=np.zeros(photo.shape[:2])
    for i in range(256):
        for j in range(256):
                # for index in range(19):
                    # if photo[i,j]==dict[index]:
                print(photo[i,j])
                        # gt[i,j]=index
    print(np.unique(gt))
    cv2.imwrite(str(i)+'_b.png',gt)
    break






