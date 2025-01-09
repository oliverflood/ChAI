# README

Everything should be run from this directory.

## getting the model

Run `python3 dump_resnet.py` to download the model and prep it for Chapel.

## Processing an image

Run `python3 ../vgg/process_image.py <path to image>` to process an image. The image will be resized to 224x224 and normalized to the imagenet mean and standard deviation. The output will be saved in a convenient format for ChAI in `../vgg/imgs/`

## Running the model

Compile the Chapel code with `chpl --fast test.chpl Resnet.chpl -M ../../src -M ../../lib -o resnet` and run it with `./resnet <path to image>`. The output will be the top 5 classes and their probabilities.

## Running the python model

Run `python3 run_resnet.py <path to image>` to run the model in python. The output will be the top 5 classes and their probabilities.
