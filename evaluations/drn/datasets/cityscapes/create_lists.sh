#!/usr/bin/env bash

#find leftImg8bit/train -maxdepth 3 -name "*_leftImg8bit.png" | sort > train_images.txt
#find /data2/gyang/TAGAN/results/test_340/images/fake_B -maxdepth 3 -name "*_B.png" | sort > val_images.txt
#find /data2/gyang/cil_net/dataset/Cityscapes/leftImg8bit/val -maxdepth 3 -name "*_leftImg8bit.png" | sort > val_images.txt

#find gtFine/train -maxdepth 3 -name "*_trainIds.png" | sort > train_labels.txt
find /data2/gyang/cil_net/dataset/Cityscapes/gtFine/val -maxdepth 3 -name "*_labelIds.png" | sort > val_labels.txt

