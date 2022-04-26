"""
Title: Cow Level Factory
Author: ESonia

Purpose: Segment cows from dataset and save segmented cows in coco-dataset format json file.

Usage: Run from the command line as such

        python3 run.py [--dataset path/to/images/dir/ (default=data/test_images/)] [--savejson path/to/answer.json (default=data/test_answer.json)] 
                    [--model [logs/]path/to/weights.h5 (default=logs/default_trained_weights/mask_rcnn_cowlevelfactory_0060.h5)] 
                    [--addon_model [AddOn/logs/]path/to/weights.h5 (default=AddOn/logs/default_trained_addon_weights/mask_rcnn_cowlevelfactory_0040.h5)]

"""

import os
import sys
import cv2
import json
import numpy as np

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import Dectectors config
from mrcnn_coco import cowDectector as cowdt
from AddOn.mrcnn_coco import estrusDectector as estdt

# Directory containing models
LOGS_DIR = "logs"
ADDON_LOGS_DIR = "AddOn/logs"
# Default directory and file paths, if not provided through the command line
DEFAULT_DATASET_DIR = "data/test_images/"
DEFAULT_SAVEJSON_PATH = "data/test_answer.json"
DEFAULT_MODEL_PATH = "default_trained_weights/mask_rcnn_cowlevelfactory_0060.h5"
DEFAULT_ADDON_MODEL_PATH = "default_trained_addon_weights/mask_rcnn_cowlevelfactory_0040.h5"



class Detector():
    def __init__(self, model_path):
        '''
        initiate Detector
        '''
        is_addon = False if not model_path==args.addon_model else True

        # Directory to initiate models
        MODEL_DIR = LOGS_DIR if not is_addon else ADDON_LOGS_DIR
        # Local path to trained weights file
        MODEL_PATH = model_path

        class InferenceConfig(cowdt.CocoConfig if not is_addon else estdt.CocoConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            # Config the minimum confidence for cowDetector and estrusDectector
            DETECTION_MIN_CONFIDENCE = 0 if not is_addon else 0

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

        # Load trained weights
        self.model.load_weights(MODEL_PATH, by_name=True)


    def detectCow(self, image):
        '''
        Run detection
        '''
        result = self.model.detect([image], verbose=0)

        return result[0]



def absolutePath(rel_path):
    '''
    Convert relative path to absolute path
    '''
    abosl_path = os.path.join(ROOT_DIR, rel_path)

    return abosl_path


def modelPath(rel_path):
    '''
    Convert relative model path to absolute model path
    '''
    # If start path is not 'logs', append 'logs'
    if os.path.commonpath([rel_path, LOGS_DIR]) is '':
        rel_path = os.path.join(LOGS_DIR, rel_path)
    abosl_path = os.path.join(ROOT_DIR, rel_path)

    return abosl_path


def addonModelPath(rel_path):
    '''
    Convert relative addon model path to absolute addon model path
    '''
    # If start path is not 'AddOn/logs', append 'AddOn/logs'
    if os.path.commonpath([rel_path, ADDON_LOGS_DIR]) is '':
        rel_path = os.path.join(ADDON_LOGS_DIR, rel_path)
    abosl_path = os.path.join(ROOT_DIR, rel_path)

    return abosl_path



def cvtBbox(roi, point=None):
    '''
    Convert roi to bbox
    roi format is [y1, x1, y2, x2]
    bbox format is [x, y, width, height]
    '''
    # Transform roi to the coco-dataset format
    bbox = [roi[1], roi[0], roi[3]-roi[1], roi[2]-roi[0]]

    # If AddOn, add the cropped amount
    if point is not None:
        bbox[0] += point[0]
        bbox[1] += point[1]

    # Cast list members np.int to float
    bbox = list(map(float, bbox))

    return bbox


def extSegmentation(mask, point=None):
    '''
    Extract contours from mask
    '''
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    # Transform contours to the coco-dataset format
    for contour in contours:
        contour = contour.squeeze(1)
        # If AddOn, add the cropped amount
        if point is not None:
            contour += point
        # Cast list members np.int to float
        contour = list(map(float, contour.ravel()))

        segmentation.append(contour)

    return segmentation



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Cow Level Factory')
    parser.add_argument('--dataset', required=False,
                        type=absolutePath, default=DEFAULT_DATASET_DIR,
                        metavar="path/to/images/dir/",
                        help='Directory of input images (default=data/test_images/)')
    parser.add_argument('--savejson', required=False,
                        type=absolutePath, default=DEFAULT_SAVEJSON_PATH,
                        metavar="path/to/answer.json",
                        help='Path to output answer .json file (default=data/test_answer.json)')
    parser.add_argument('--model', required=False,
                        type=modelPath, default=DEFAULT_MODEL_PATH,
                        metavar="[logs/]path/to/weights.h5",
                        help="Path to weights .h5 file (default=logs/default_trained_weights/mask_rcnn_cowlevelfactory_0060.h5)")
    parser.add_argument('--addon_model', required=False,
                        type=addonModelPath, default=DEFAULT_ADDON_MODEL_PATH,
                        metavar="[AddOn/logs/]path/to/weights.h5",
                        help="Path to addon_weights .h5 file (default=AddOn/logs/default_trained_addon_weights/mask_rcnn_cowlevelfactory_0040.h5)")
    args = parser.parse_args()
    print("Dataset: ", args.dataset)
    print("Savejson: ", args.savejson)
    print("Model: ", args.model)
    print("AddOn_Model: ", args.addon_model)


    # Declare json Objects
    json_answer = {"categories":[], "images":[], "annotations":[]}
    json_cats = json_answer["categories"]
    json_imgs = json_answer["images"]
    json_anns = json_answer["annotations"]


    # Update json categories
    json_cats.extend([{"id":1, "name":"normal_cow"}, {"id":2, "name":"estrus_cow"}])


    # Directory of images
    DATASET_DIR = args.dataset
    # Get image file names from the images folder
    file_names = next(os.walk(DATASET_DIR))[2]

    # Update json images
    images = []
    img_count = len(file_names)
    for i in range(img_count):
        img = cv2.imread("{}/{}".format(DATASET_DIR, file_names[i]))
        h, w, c = img.shape
        images.append({"id":i, "file_name":file_names[i], "width":w, "height":h})
    json_imgs.extend(images)

    
    # Update json annotations
    annotations = []
    anno_id = 0
    # Create Mask-RCNN Models and Load Trained Weights
    cow_detector = Detector(args.model)
    est_detector = Detector(args.addon_model)

    total_score_sum = 0.000
    prev_anno_count = 0
    # Detect cows and create annotations
    for i in range(img_count):
        # Read a Image from the images folder
        image = cv2.cvtColor(cv2.imread(os.path.join(DATASET_DIR, file_names[i])), cv2.COLOR_BGR2RGB)
        # Detect cows from the image
        result = cow_detector.detectCow(image)
        cow_count = len(result["rois"])
        # Transpose the 3D masks array from height[width[cows[, ], ], ] to cows[height[width[, ], ], ]
        masks = result["masks"].transpose(2, 0, 1)
        masks = masks.astype('uint8')

        score_sum = 0.000
        # Create annotations for each cow
        for ci in range(cow_count):
            if result["class_ids"][ci]==1:
                # Add creaded annotation
                annotations.append({"id":anno_id, "image_id":images[i]["id"], "bbox":cvtBbox(result["rois"][ci]), 
                                    "segmentation":extSegmentation(masks[ci]), "category_id":int(result["class_ids"][ci]), 
                                    "conf":float(result["scores"][ci])})
                anno_id += 1

                score_sum += result["scores"][ci]
            # If the found cow ID is 2, Run Estrus Detector
            elif result["class_ids"][ci]==2:
                # roi format is [y1, x1, y2, x2]
                roi = result["rois"][ci]
                # Crop the image except for area containing a estrus cow
                # cv2.cropping format is [y1:y2, x1:x2]
                addon_image = image[roi[0]:roi[2], roi[1]:roi[3]]
                # Detect the estrus cow from the cropped image
                addon_result = est_detector.detectCow(addon_image)
                est_count = len(addon_result["rois"])
                # Transpose the 3D masks array from height[width[cows[, ], ], ] to cows[height[width[, ], ], ]
                addon_masks = addon_result["masks"].transpose(2, 0, 1)
                addon_masks = addon_masks.astype('uint8')

                cropped_point = [roi[1], roi[0]]
                # Add creaded annotation
                for ei in range(est_count):
                    annotations.append({"id":anno_id, "image_id":images[i]["id"], "bbox":cvtBbox(addon_result["rois"][ei], cropped_point), 
                                        "segmentation":extSegmentation(addon_masks[ei], cropped_point), "category_id":int(result["class_ids"][ci]), 
                                        "conf":float(addon_result["scores"][ei])})
                    anno_id += 1

                    score_sum += addon_result["scores"][ei]
            else:
                raise Exception("exceeded number of classes")

        # Calculate average scores
        score_avg = round((score_sum/(anno_id-prev_anno_count)), 3)
        total_score_sum += score_avg
        total_score_avg = round((total_score_sum/(i+1)), 3)
        prev_anno_count = anno_id
        # Print progress bar
        progress = int(np.trunc((i+1)/img_count*30))
        print("%4d/%4d ["%(i+1, img_count) + ''.join('=' for p in range(progress-1)) + ''.join('>' if progress>0 else '') + 
              '.'*(30-progress) + '] - score_avg: %.03f - total_score_avg: %.03f'%(score_avg, total_score_avg))
    json_anns.extend(annotations)


    # Path of json for write
    SAVEJSON_PATH = args.savejson

    # Save json_answer to SAVEJSON_PATH
    with open(SAVEJSON_PATH, 'w') as json_file:
        json.dump(json_answer, json_file)

    print("Save json_answer to \"{}\".".format(SAVEJSON_PATH))
