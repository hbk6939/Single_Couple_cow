import json
from PIL import Image

# Json
JSON_DIR = './addon_train_answer.json'
with open(JSON_DIR, 'r') as json_file:
    json_obj = json.load(json_file)

obj_cat= json_obj['categories']
obj_img= json_obj['images']
obj_ann= json_obj['annotations']


imgIds = []
bbox = {}
for ann in obj_ann:
    bbox.update({ann['image_id']:ann['bbox']})


for img in obj_img:
    image = Image.open('./train_images/'+img['file_name'])
    croppedImage = image.crop((bbox[img['id']][0], bbox[img['id']][1], bbox[img['id']][0]+bbox[img['id']][2], bbox[img['id']][1]+bbox[img['id']][3]))
    croppedImage.save('./addon_train_images/'+img['file_name'])

