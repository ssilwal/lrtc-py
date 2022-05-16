import bcd
import numpy as np
from PIL import Image
import csv
import os

dir_name = 'art'
out_dir_name = 'art_results'

if(not os.path.isdir(out_dir_name)):
	os.mkdir(out_dir_name)

img_list = os.listdir(dir_name)
for img_name in img_list:
  img = Image.open(os.path.join(dir_name,img_name))
  X = np.asarray(img)
  a = np.array([1,1,1])
  b = np.array([1,1,1])
  gamma = [100,100,0]
  print('begin running for image:' + img_name)
  itr =200
  result,errors = bcd.lrtc(X,a,b,gamma,max_itr=itr)
  print('completed ' + str(itr) + " iterations")
  res_img = Image.fromarray(result.astype(np.uint8))
  res_img.save(os.path.join(out_dir_name,img_name))

  # see: https://pythonguides.com/python-write-a-list-to-csv/
  with open(os.path.join(out_dir_name,img_name +'_log.txt'),'w') as log_file:
    write=csv.writer(log_file)
    write.writerows(errors)

