from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = '../data/train_images'
annFile = './annotations/formatted_train_answer.json'

# initialize COCO api for instance annotations
coco = COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

#nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}\n'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms = [[]])    #catNms가 모두 포함된 사진들 가져오기. ex) catNms = ['single_cow', 'couple_cow']
imgIds = coco.getImgIds(catIds = catIds)
#imgIds = coco.getImgIds(imgIds = [41])    #번호 선택해서 가져오기.
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
print('selected image info: \n{}\n'.format(img))

# load and display instance annotations
I = io.imread('%s/%s'%(dataDir,img['file_name']))
plt.imshow(I); plt.axis('off')
annIds = coco.getAnnIds(imgIds = img['id'], catIds = catIds, iscrowd = None)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)
plt.show()