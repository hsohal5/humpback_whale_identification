from sklearn.model_selection import train_test_split
from keras.utils import Sequence
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from keras import backend as K
from keras.preprocessing.image import array_to_img
from numpy.linalg import inv as mat_inv

with open('../input/humpback-whale-identification-fluke-location/cropping.txt', 'rt') as f: data = f.read().split('\n')[:-1]
data = [line.split(',') for line in data]
data = [(p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) for p,*coord in data]
img_shape  = (128,128,1)

train, val = train_test_split(data, test_size=200, random_state=1)
train += train
train += train
train += train
val_a = np.zeros((len(val),)+img_shape,dtype=K.floatx())
val_b = np.zeros((len(val),4),dtype=K.floatx()) 
def read_for_validation(p):
    x  = read_array(p)
    t  = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    t  = center_transform(t, x.shape)
    x  = transform_img(x, t)
    x -= np.mean(x, keepdims=True)
    x /= np.std(x, keepdims=True) + K.epsilon()
    return x,t 

class dataprocess(Sequence):
    def __init__(self, batch_size=32):
        super(dataprocess, self).__init__()
        self.batch_size = batch_size
    def __getitem__(self, index):
        start = self.batch_size*index;
        end   = min(len(train), start + self.batch_size)
        size  = end - start
        a     = np.zeros((size,) + img_shape, dtype=K.floatx())
        b     = np.zeros((size,4), dtype=K.floatx())
        for i,(p,coords) in enumerate(train[start:end]):
            img,trans   = read_for_training(p)
            coords      = coord_transform(coords, mat_inv(trans))
            x0,y0,x1,y1 = bounding_rectangle(coords)
            a[i,:,:,:]  = img
            b[i,0]      = x0
            b[i,1]      = y0
            b[i,2]      = x1
            b[i,3]      = y1
        return a,b
    def __len__(self):
        return (len(train) + self.batch_size - 1)//self.batch_size

random.seed(1)
a, b = dataprocess(batch_size=5)[1]
img  = array_to_img(a[0])
img  = img.convert('RGB')
draw = Draw(img)
draw.rectangle(b[0], outline='red')

model = Sequential()
model.add(Conv2D(64, (9, 9), strides = (2, 2), name = 'conv0', input_shape = (128, 128, 1)))
model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))
# model.add(Dropout(0.25))
#model.add(MaxPooling2D((2, 2), strides = (2,2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (3,3), name="conv1"))
model.add(BatchNormalization(axis = 3, name = 'bn1'))
model.add(Activation('relu'))
# model.add(Dropout(0.25))
#model.add(MaxPooling2D((2, 2), strides = (3,3), name='max_pool2'))
model.add(Conv2D(64, (3, 3), name="conv2"))
model.add(BatchNormalization(axis = 3, name = 'bn2'))
model.add(Activation('relu'))
# model.add(Dropout(0.25))
# model.add(AveragePooling2D((3, 3), name='avg_pool'))
model.add(Conv2D(64, (3, 3), name="conv3"))
model.add(BatchNormalization(axis = 3, name = 'bn3'))
model.add(Activation('relu'))
# model.add(Dropout(0.25))
model.add(MaxPooling2D((1, 1), name='max_pool4'))
model.add(Flatten())
model.add(Dense(16, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))
model.compile(loss='mean_squared_error', optimizer="adam")
model.summary()
model.fit_generator(
        TrainingData(), epochs=50, max_queue_size=12, workers=4, verbose=1,
        validation_data=(val_a, val_b),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=9, min_delta=0.1, verbose=1)
        ])
model.evaluate(val_a, val_b, verbose=0)
p2bb = {}
img_train_path = os.path.abspath('C:/Users/seera/Desktop/ML/train')
train_paths = [img for img in os.listdir(img_train_path)]
img_test_path = os.path.abspath('C:/Users/seera/Desktop/ML/test')
test_paths = [img for img in os.listdir(img_test_path)]
csv_train_path = os.path.abspath('C:/Users/seera/Desktop/ML/train.csv')
df = pd.read_csv(csv_train_path)
bbox_df = pd.DataFrame(columns=['Image','x0','y0','x1','y1']).set_index('Image')
def make_bbox(p):
    raw = read_array(p)
    width, height = raw.shape[1], raw.shape[0]
    img,trans         = read_for_validation(raw)
    a                 = np.expand_dims(img, axis=0)
    x0, y0, x1, y1    = model.predict(a).squeeze()
    (u0, v0),(u1, v1) = coord_transform([(x0,y0),(x1,y1)], trans)
    bbox = [max(u0,0), max(v0,0), min(u1,width), min(v1,height)]
    if bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
        bbox = [0,0,width,height]
    return bbox
for img in tqdm(train_paths):
    bbox_df.loc[img] = make_bbox(os.path.join(img_train_path,img))
for img in tqdm(test_paths):
    bbox_df.loc[img] = make_bbox(os.path.join(img_test_path,img))
#bbox_df.to_csv("bounding_boxes.csv")
train_paths = pd.DataFrame((os.path.join(img_train_path, img),img) for img in os.listdir(img_train_path))
train_paths.columns = ['path','img']
test_paths = pd.DataFrame((os.path.join(img_test_path, img),img) for img in os.listdir(img_test_path))
test_paths.columns = ['path','img']
def crop(img_path,img):
    """
    :param img: path to image
    """
    main_img = image.load_img(img_path)
    img_crop = main_img.crop(tuple(bbox_df.loc[img,:]))
    return img_crop

for row in train_paths.itertuples(index=True, name='Pandas'):
    cpimg = crop(getattr(row, 'path'), getattr(row, 'img'))
    img=getattr(row, 'img')
    dircp = "C:\\Users\\seera\\Desktop\\ML\\traincrop\\"
    pt = os.path.join(dircp,img)

for row in test_paths.itertuples(index=True, name='Pandas'):
    cpimg = crop(getattr(row, 'path'), getattr(row, 'img'))
    img=getattr(row, 'img')
    dircp = "C:\\Users\\seera\\Desktop\\ML\\testcrop\\"
    pt = os.path.join(dircp,img)


