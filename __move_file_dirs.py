#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 19:52:27 2017

@author: derek
"""

move_dirs = glob.glob("/media/derek/disk1/kaggle_ndsb2017/luna_raw/subset*")
for i in move_dirs[1:]:
    files_ = os.listdir(i)
    for f in files_:
        shutil.copyfile(i + "/" + f,end_dir+f)


end_dir = "/media/derek/disk1/kaggle_ndsb2017/resources/_luna16_mhd/"


shutil.copyfile(i,end_dir+i)


dirs = glob.glob("/media/derek/disk1/kaggle_ndsb2017/resources/luna16_annotations/*")

for d in dirs:
    if os.path.isdir(d):
        for f in os.listdir(d):
            folder = d.split('/')[-1]
            shutil.copyfile(d + "/" + f,"/media/derek/disk1/kaggle_ndsb2017/resources/_luna16_xml/"+folder+"_"+f)
            print((f))




