# run.py
-Purpose: Segment cows from dataset and save segmented cows in coco-dataset format json file.

-Usage: Run from the command line as such

        python3 run.py [--dataset path/to/images/dir/ (default=data/test_images/)] [--savejson path/to/answer.json (default=data/test_answer.json)] 
                    [--model [logs/]path/to/weights.h5 (default=logs/default_trained_weights/mask_rcnn_cowlevelfactory_0060.h5)] 
                    [--addon_model [AddOn/logs/]path/to/weights.h5 (default=AddOn/logs/default_trained_addon_weights/mask_rcnn_cowlevelfactory_0040.h5)]



# viewer.py
-Purpose: Display the inference results.

-Usage: Run from the command line as such

        python3 viewer.py [--id <image id | -1 (random id)> (default=-1)] 
                        [--dataset path/to/images/dir/ (default=data/test_images)] [--json path/to/answer.json (default=data/test_answer.json)]
                        [--mask <True | False> (default=True)] [--bbox <True | False> (default=True)] [--label <True | False> (default=True)] 



# mrcnn_coco/cowDectector.py
# AddOn/mrcnn_coco/estrusDectector.py
-Location: File and Directory must be in this location form for run

	# Directory of dataset
		/path/to/data/{train | val}_images/

	# Json file in coco-dataset format
		/path/to/data/{train | val}_answer.json

-Usage: Run from the command line as such:

    # Train a new model starting from pre-trained COCO weights
		python3 coco.py train --dataset=/path/to/data/ --model=coco

    # Continue training a model that you had trained earlier
		python3 coco.py train --dataset=/path/to/data/ --model=/path/to/weights.h5

    # Continue training the last model you trained
		python3 coco.py train --dataset=/path/to/data/ --model=last

    # Run COCO evaluatoin on the last model you trained
		python3 coco.py evaluate --dataset=/path/to/data/ --model=last



# mrcnn_coco/inspect_data.ipynb
# mrcnn_coco/inspect_model.ipynb
# mrcnn_coco/inspect_weights.ipynb
-Reference: https://github.com/matterport/Mask_RCNN