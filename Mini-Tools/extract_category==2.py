import json
import numpy as np

OUT_DIR = './ext_p1_train_answer.json'
# Json
JSON_DIR = './p1_train_answer.json'
with open(JSON_DIR, 'r') as json_file:
    json_obj = json.load(json_file)

obj_cat= json_obj['categories']
obj_img= json_obj['images']
obj_ann= json_obj['annotations']


imgIds = []
newAnns = []
for ann in obj_ann:
    if ann['category_id'] == 2:
        imgIds.append(ann['image_id'])
        ann['category_id'] = 1
        newAnns.append(ann)

newImgs = []
for img in obj_img:
    img['file_name'] = 'addon_' + img['file_name']
    if img['id'] in imgIds:
        newImgs.append(img)

newCats = []
obj_cat[0]['name'] = 'estrus_cow'
newCats.append(obj_cat[0])

newJson = {"categories":newCats, "images":newImgs, "annotations":newAnns}
with open(OUT_DIR, 'w') as f:
    json.dump(newJson, f)
