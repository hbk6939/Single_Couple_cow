import json
import numpy as np
import re

OUT_DIR = './p2_addon_train_answer.json'
# Json
JSON_DIR = './p1_addon_train_answer.json'
with open(JSON_DIR, 'r') as json_file:
    json_obj = json.load(json_file)

obj_cat= json_obj['categories']
obj_img= json_obj['images']
obj_ann= json_obj['annotations']


imgIds = {}
for img in obj_img:
    imgIds.update({img['id']:int(re.sub(r'[^0-9]', '', img['file_name']))})
    img['id'] = imgIds[img['id']]

annId = 1
for ann in obj_ann:
    ann['id'] = annId
    annId+=1
    ann['image_id'] = imgIds[ann['image_id']]

newJson = {"categories":obj_cat, "images":obj_img, "annotations":obj_ann}
with open(OUT_DIR, 'w') as f:
    json.dump(newJson, f)
