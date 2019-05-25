import pandas as pd
from os.path import isfile
from PIL import Image as pil_image
import numpy as np
from math import sqrt
from imagehash import phash
#from imagehash import phash
# Read the dataset description
data= pd.read_csv("C:\\Users\\solan\\Downloads\\Subjects\\Winter\\Machine Learning\\Project\\humpback-whale-identification\\train.csv")
data= dict([(p,w) for _,p,w in data.to_records()])
submit = [p for _,p,_ in pd.read_csv("C:\\Users\\solan\\Downloads\\Subjects\\Winter\\Machine Learning\\Project\\humpback-whale-identification\\sample_submission.csv").to_records()]
join   = list(data.keys()) + submit
len(data),len(submit),len(join),list(data.items())[:5],submit[:5] #not same as the kernel
# size of each image
def address(p):
    if isfile('C:\\Users\\solan\\Downloads\\Subjects\\Winter\\Machine Learning\\Project\\humpback-whale-identification\\train\\' + p): 
        return 'C:\\Users\\solan\\Downloads\\Subjects\\Winter\\Machine Learning\\Project\\humpback-whale-identification\\train\\' + p
    if isfile('C:\\Users\\solan\\Downloads\\Subjects\\Winter\\Machine Learning\\Project\\humpback-whale-identification\\test\\' + p): 
        return 'C:\\Users\\solan\\Downloads\\Subjects\\Winter\\Machine Learning\\Project\\humpback-whale-identification\\test\\' + p
    return p
p2size = {}
for p in join:
   # print(p)
    use = pil_image.open(address(p))
    size = use.size
    #print(size)
    p2size[p] = size
    
    #print(pil_image.open(expand_path(p)))
    #print('hello')
#print(p2size.items())
#len(p2size),list(p2size.items())[:5]

def compare(h1,h2):
    for p1 in h2ps[h1]:
        #print(type(p1))
        for p2 in h2ps[h2]:
            print(p2,h2)
            image1 =  pil_image.open(address(p1))
            print(image1)
            image2 =  pil_image.open(address(p2))
            if image1.mode != image2.mode or image1.size != image2.size: 
                return False
            f1 = np.array(image1)
            f1 = f1 - f1.mean()
            f1 = f1/sqrt((f1**2).mean())
            f2 = np.array(image2)
            f2 = f2 - f2.mean()
            f2 = f2/sqrt((f2**2).mean())
            f  = ((f1 - f2)**2).mean()
            if f > 0.1: 
                return False
    return True
#if isfile('C:\\Users\\solan\\Downloads\\Subjects\\Winter\\Machine Learning\\Project\\humpback-whale-identification\\humpback-whale-identification-model-files\\p2h.pickle'):
 #   with open('C:\\Users\\solan\\Downloads\\Subjects\\Winter\\Machine Learning\\Project\\humpback-whale-identification\\humpback-whale-identification-model-files\\p2h.pickle','rb') as f:
  #      p2h = pickle.load(f)
#else:
    # Compute phash for each image in the training and test set.
p2h = {}
for p in join:
    img    = pil_image.open(address(p))
    h      = phash(img)
    p2h[p] = h
    print(h)
 # Find all images associated with a given phash value.
#print(p2h)
global h2ps
h2ps = {}
for p,h in p2h.items():
    if h not in h2ps:
        h2ps[h] = []
        if p not in h2ps[h]:
            h2ps[h].append(p)
 # Find all distinct phash values
hs = list(h2ps.keys())
#print(len(hs))
#print(hs)
# If the images are close enough, associate the two phash values 
h2h = {}
#print(h2ps)
for i,h1 in enumerate(hs):
    print(h1)
    for h2 in hs[:i]:
         if h1-h2 <= 6 and compare(h1, h2):
            q1 = str(h1)
            q2 = str(h2)
            if q1 < q2: 
                q1,q2 = q2,q1
            h2h[q1] = q2
# Group together images with equivalent phash, and replace by string format of phash (faster and more readable)
    for p,h in p2h.items():
        h = str(h)
        if h in h2h: h = h2h[h]
        p2h[p] = h

#len(p2h), list(p2h.items())[:5]
# For each image id, determine the list of pictures
h2ps = {}
for p,h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)
# 25460 images use only 20913 distinct image ids.
len(h2ps),list(h2ps.items())[:5]
# For each images id, select the prefered image
def preferred(ps):
    if len(ps) == 1: return ps[0]
    prefer_p = ps[0]
    prefer_s = p2size[prefer_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0]*s[1] > prefer_s[0]*prefer_s[1]: # Select the image with highest resolution
            prefer_p = p
            prefer_s = s
    return prefer_p

h2p = {}
for h,ps in h2ps.items(): 
    h2p[h] = preferred(ps)
#len(h2p),list(h2p.items())[:5]