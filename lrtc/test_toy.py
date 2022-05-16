import bcd
import numpy as np
from PIL import Image
import csv

img = Image.open('testImg.png')
X = np.asarray(img)
gamma = [100,100,0]
result,errors = bcd.lrtc(X,gamma,max_itr=20)

result_name = 'test_result.jpg'
import pdb
pdb.set_trace()
res_img = Image.fromarray(result.astype(np.uint8))
res_img.save(result_name)

# see: https://pythonguides.com/python-write-a-list-to-csv/
with open('test_log.txt','w') as log_file:
  write=csv.writer(log_file)
  write.writerows(errors)

