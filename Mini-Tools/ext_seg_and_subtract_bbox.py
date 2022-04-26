import json
import numpy as np

OUT_DIR = './addon_p1_train_answer.json'

# Json1
JSON_DIR = './ext_p1_train_answer.json'
with open(JSON_DIR, 'r') as json_file:
    json_obj = json.load(json_file)

obj_cat= json_obj['categories']
obj_img= json_obj['images']
obj_ann= json_obj['annotations']


# Json2
JSON_DIR2 = './ext_p0_train_answer.json'
with open(JSON_DIR2, 'r') as json_file:
    json_obj2 = json.load(json_file)
    
#obj_cat2 = json_obj2['categories']
#obj_img2 = json_obj2['images']
obj_ann2 = json_obj2['annotations']



# ann2seg = {Image Id : segmentation}
ann2seg = {}
for ann2 in obj_ann2:
    ann2seg.update({ann2['image_id']:ann2['segmentation']})

# Json1 Image 크기 - bbox(width, height)
# json1<= json2 segmentation 정보
# segmentation에 bbox(x, y) 값 더한 후 저장
for ann in obj_ann:
    for img in obj_img:
        if img['id'] == ann['image_id']:
            img['width'] = ann['bbox'][2]
            img['height'] = ann['bbox'][3]
            break
    if ann['image_id'] in ann2seg:
        ann['segmentation']= ann2seg[ann['image_id']]
    i= 0
    for annSeg in ann['segmentation']:
        annSeg = np.reshape(annSeg, (int(len(annSeg)/2), 2))- ann['bbox'][:2]
        for j in range(len(annSeg)):
            if annSeg[j][0]> ann['bbox'][2]:
                annSeg[j][0] = ann['bbox'][2]
            if annSeg[j][1]> ann['bbox'][3]:
                annSeg[j][1] = ann['bbox'][3]
        annSeg = annSeg.ravel()
        newSeg = []
        for j in range(len(annSeg)):
            if annSeg[j]<0:
                newSeg.append(round(0.0, 1))
            else:
                newSeg.append(round(annSeg[j], 1))
        ann['segmentation'][i] = newSeg
        i+= 1
    ann.pop('area')

newJson = {"categories":obj_cat, "images":obj_img, "annotations":obj_ann}
with open(OUT_DIR, 'w') as f:
    json.dump(newJson, f)

