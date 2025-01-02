import os
from PIL import Image
import numpy as np
from tqdm import tqdm

rd = '/home/mail/2017m7/m730602003/dataset/shadow/SBU-shadow/SBUTrain4KRecoveredSmall/labels'
files = os.listdir(rd)
for f in tqdm(files):
    p = os.path.join(rd, f)
    img = np.sum(np.sum((np.array(Image.open(p))>0)))

    if img<=0:
        print(f)