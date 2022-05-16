import bcd
import numpy as np
from PIL import Image


img = Image.open('testImg.png')
X = np.asarray(img)
gamma = [100,100,0]
result = bcd.lrtc(X,gamma,max_itr=20)

result_name = 'test_result.jpg'
import pdb
pdb.set_trace()
res_img = Image.fromarray((result/256).astype(np.uint8))
res_img.save(result_name)

