"""
Title: COCO Dataset Viewer
Author: ESonia

Purpose: Display the inference results.

Usage: Run from the command line as such

        python3 viewer.py [--id <image id | -1 (random id)> (default=-1)] 
                        [--dataset path/to/images/dir/ (default=data/test_images/)] [--json path/to/answer.json (default=data/test_answer.json)]
                        [--mask <True | False> (default=True)] [--bbox <True | False> (default=True)] [--label <True | False> (default=True)] 

"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import visualize

# Default directory and file path, if not provided through the command line
DEFAULT_DATASET_DIR = "data/test_images/"
DEFAULT_SAVE_PATH = "data/test_answer.json"



class COCO(COCO):
    def showAnns(self, annotations, draw_mask=True, draw_bbox=True, draw_label=True):
        '''
        Display the specified annotations
        '''
        from matplotlib.collections import PatchCollection
        from matplotlib.patches import Polygon
        from matplotlib.patches import Rectangle

        if len(annotations) == 0:
            return 0

        ax = plt.gca()
        ax.set_autoscale_on(False)
        polygons = []
        rectangles = []
        poly_color = []
        rect_color = []

        for anno in annotations:
            color = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
            # Mask
            if draw_mask:
                for seg in anno['segmentation']:
                    poly = np.array(seg).reshape((int(len(seg)/2), 2))
                    polygons.append(Polygon(poly))
                    poly_color.append(color)
            # Bounding box
            if draw_bbox and 'bbox' in anno:
                rect_x, rect_y, rect_w, rect_h = anno['bbox']
                rectangles.append(Rectangle((rect_x, rect_y), rect_w, rect_h))
                rect_color.append(color)
                # Label
                if draw_label and 'conf' in anno:
                        score = anno['conf']
                        label = self.loadCats(anno['category_id'])[0]['name']
                        caption = "{} {:.3f}".format(label, score) if score else label
                        ax.text(rect_x+4, rect_y+13, caption, color='w', size=11, backgroundcolor='none')

            p = PatchCollection(polygons, facecolor=poly_color, linewidths=0, alpha=0.1)
            ax.add_collection(p)
            p = PatchCollection(polygons, facecolor='none', edgecolors=poly_color, linewidths=1, alpha=0.7)
            ax.add_collection(p)
            p = PatchCollection(rectangles, linestyle="dashed", facecolor='none', edgecolor=rect_color, 
                                    linewidth=1, alpha=0.7)
            ax.add_collection(p)



def absolutePath(rel_path):
    '''
    Convert relative path to absolute path
    '''
    abosl_path = os.path.join(ROOT_DIR, rel_path)

    return abosl_path


def boolean(var):
    '''
    Return variable to bool
    '''
    if isinstance(var, bool):
        return var
    if var.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    elif var.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected')



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run Cow Level Factory')
    parser.add_argument('--id', required=False,
                        type=int, default=-1,
                        metavar="<image id | -1 (random id)>",
                        help='Image ID to read. if the value is -1, read a random image (default=-1)')
    parser.add_argument('--dataset', required=False,
                        type=absolutePath, default=DEFAULT_DATASET_DIR,
                        metavar="path/to/images/dir/",
                        help='Directory of images (default=data/test_images/)')
    parser.add_argument('--json', required=False,
                        type=absolutePath, default=DEFAULT_SAVE_PATH,
                        metavar="path/to/answer.json",
                        help='Path to answer .json file (default=data/test_answer.json)')
    parser.add_argument('--mask', required=False,
                        type=boolean, default=True,
                        metavar="<True | False>",
                        help='Draw masks on the image (default=True)')
    parser.add_argument('--bbox', required=False,
                        type=boolean, default=True,
                        metavar="<True | False>",
                        help='Draw bound boxes on the image (default=True)')
    parser.add_argument('--label', required=False,
                        type=boolean, default=True,
                        metavar="<True | False>",
                        help='Draw labels on the image. if the bbox is False, not draw (default=True)')
    args = parser.parse_args()
    print("id: ", args.id if args.id>=0 else "(random)")
    print("Dataset: ", args.dataset)
    print("Json: ", args.json)
    print("Mask: ", args.mask)
    print("Bbox: ", args.bbox)
    print("label: ", args.label)

    # Directory of images
    DATASET_DIR = args.dataset
    # Path of json for read
    JSON_PATH = args.json

    # Initialize COCO api for instance annotations
    coco = COCO(JSON_PATH)

    # Display categories
    categories = coco.loadCats(coco.getCatIds())
    cat_names = [cat['name'] for cat in categories]
    print('Categories: \n{}'.format(cat_names))

    # Read a image corresponding to given image ID. If the given was -1, read a random image
    img_ids = coco.getImgIds()
    sel_Id = args.id if args.id>=0 else np.random.randint(0,len(img_ids))
    sel_img = coco.loadImgs(img_ids[sel_Id])[0]
    print('Selected image info: \n{}'.format(sel_img))

    # Load and display instance annotations
    image = cv2.cvtColor(cv2.imread(os.path.join(DATASET_DIR, sel_img['file_name'])), cv2.COLOR_BGR2RGB)
    plt.imshow(image); plt.axis('off')
    anno_ids = coco.getAnnIds(imgIds=sel_img['id'])
    annotations = coco.loadAnns(anno_ids)
    coco.showAnns(annotations, draw_mask=args.mask, draw_bbox=args.bbox, draw_label=args.label)
    plt.show()
