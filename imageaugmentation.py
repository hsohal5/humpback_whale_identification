import os
import pandas as pd
import csv
import shutil
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image as pil_image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
img_train_path = os.path.abspath('C:/Users/seera/Desktop/ML/traincrop')
#img_test_path = os.path.abspath('C:/Users/seera/Desktop/ML/testcrop')
csv_train_path = os.path.abspath('C:/Users/seera/Desktop/ML/train.csv')
df = pd.read_csv(csv_train_path)
one = df.groupby("Id").filter(lambda x: len(x) <= 15)
two = df.groupby("Id").filter(lambda x: len(x) > 15)
datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
src_dir = r"C:\Users\seera\Desktop\ML\traincrop"
dst_dir = r"C:\Users\seera\Desktop\ML\augimg2"
dst_dir1 = r"C:\Users\seera\Desktop\ML\nonaugimg"

df1 = dict([(p,w) for _,p,w in one.to_records()])
df1 = dict([(p,w) for _,p,w in two.to_records()])

for p,w in df1.items():
    img= os.path.join(str(src_dir), str(p))
    shutil.copy(img,dst_dir)
for p,w in df2.items():
    img= os.path.join(str(src_dir), str(p))
    shutil.copy(img,dst_dir1)
with open(r'C:\Users\seera\Desktop\ML\augimg.csv', mode='w') as whale:
    fieldnames = ['Image', 'Id']
    writer = csv.DictWriter(whale, fieldnames=fieldnames)
    writer.writeheader()
    
    for p,w in df1.items():
        im= os.path.join(str(dst_dir), str(p))
        im = load_img(im)  
        x = img_to_array(im)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        i = 0
        j= 0
#         for batch in datagen.flow(x, batch_size=1,
#                               save_to_dir=r"C:\Users\seera\Desktop\ML\augimg2", save_prefix=p, save_format='jpeg'):
        for i in range(0,16):
            writer.writerow({'Image':p+'_'+str(i)+'.jpeg' , 'Id': w })
            print(str(j))
            i += 1
            j += 1
            if i > 16:
                break 

