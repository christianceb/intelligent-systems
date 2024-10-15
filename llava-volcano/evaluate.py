# LLaVA imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Imports that are either shared with LLaVa or used for preprocessing
from datetime import datetime
from typing import List
from PIL import Image, PngImagePlugin
from random import random
import csv
import json
import os
import time
import logging

# Load LLaVa into a variable
model_path = "liuhaotian/llava-v1.5-7b"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_4bit=True, # Try running this someday with it not set. It's only defined because an error perists when you run the script
)

"""
Options
"""
SCALE_MAX_DIMENSION = 1024
SAMPLE_DATASET_PERCENT = 0.001
SKIP_IMAGE_RESIZE = True # Skip the expensive process of resizing

# You will want it that both values below are equal, especially if you are recreating the test dataset.
SKIP_IMAGE_PREPROCESSING = False
USE_EXISTING_SPLIT_DATASET = False

# When not in preprocessing, leave this string to `""`
OVERRIDE_FOLDER = ""

"""
Additional options that you likely wont have to change.
"""
DATASET_PATH = "../compare_ds"
IMAGES_PATH = DATASET_PATH + "/images"
ANNOTATIONS_PATH = DATASET_PATH + "/annotations"
ANNOTATION_SOURCES = [ANNOTATIONS_PATH + "/class_train_annotation.json", ANNOTATIONS_PATH + "/class_val_annotation.json"]
POSTPROCESS_PATH = OVERRIDE_FOLDER + "postprocessed_images"
GUESS_IMAGE_FILE_EXTENSION = ".png"
TARGET_IMAGE_FILE_EXTENSION = ".jpg"
MERGED_IMAGE_GUTTER_SIZE = 20
EXISTING_SPLIT_DATASET_PATH = OVERRIDE_FOLDER + "splitted_annotations.json"
OUTPUT_TABULATED_PREDICT_RESULTS_JSON_PATH = OVERRIDE_FOLDER + "tabulated_predict_results.json"
OUTPUT_TABULATED_PREDICT_RESULTS_CSV_PATH = OVERRIDE_FOLDER + "tabulated_predict_results.csv"

def initialise_logger():
    """
    Logger for debug tracing. Feel free to ignore me!
    """
    LOG_FOLDER = OVERRIDE_FOLDER + "logs"
    LOG_FILE = LOG_FOLDER + "/llava_" + datetime.now().strftime('%Y-%m-%d') + ".log"

    os.makedirs(LOG_FOLDER, exist_ok=True)

    logging.basicConfig(filename=LOG_FILE, level=logging.INFO)

initialise_logger()

def log_info(message: str, context: dict = {}):
    """
    Use the logger to log some info `log.info()`
    """
    log = logging.getLogger()

    if bool(context):
        log.info(message + " " +json.dumps(context))
    else:
        log.info(message)


def save_predict_results(results):
    """
    Saves the prediction results in both JSON and CSV
    """

    # JSON
    with open(OUTPUT_TABULATED_PREDICT_RESULTS_JSON_PATH, "w") as outfile: 
        json.dump(results, outfile)

    # CSV
    keys = results[0].keys()

    with open(OUTPUT_TABULATED_PREDICT_RESULTS_CSV_PATH, 'w', newline='') as csv_file:
        dict_writer = csv.DictWriter(csv_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)


def start_predicting(annotations):
    """
    Predict the annotation and attributes
    """

    predict_results = []
    annotations_size = len(annotations)
    attributes_size = 0
    correct_count = 0
    incorrect_count = 0

    log_info("About to start predicting", { "annotations": annotations_size })

    for index_annotation, annotation in enumerate(annotations):
        log_info("Running annotation", { "index": index_annotation, "percent": int((index_annotation / annotations_size) * 100) })

        for index_attribute, attribute in enumerate(annotation['attributes']):
            attributes_size += 1
            start_time = current_milli_time()
            
            # We need to make our prompt coherent for the model to understand what we want
            prompt = preprocess_prompt(attribute)

            log_info("About to predict attribute in annotation", {
                "annotation_index": index_annotation,
                "attribute": f"{index_attribute}/{len(annotation['attributes'])}",
            })

            # Have the model make a prediction
            prediction = predict(annotation['filepath'], prompt)

            # Based on the text produced from the prediction, evaluate if the model correctly guessed it or not
            evaluated_prediction = evaluate_prediction(prediction, attribute['answer'])
            
            if evaluated_prediction:
                correct_count += 1
            else:
                incorrect_count += 1

            # Stash results
            predict_results.append({
                "name": annotation['name'],
                "correctly_predicted": evaluated_prediction,
                "prompt": prompt,
                "key": attribute['key'],
                "result": prediction,
                "answer": attribute['answer'],
                "time_elapsed": current_milli_time() - start_time
            })

    return predict_results, annotations_size, correct_count, incorrect_count, attributes_size

def current_milli_time():
    return round(time.time() * 1000)

def preprocess_prompt(attribute) -> str:
    """
    Preprocess the prompt to make it coherent. You can change or add more lines or context to affect the result of the prediction.
    """
    prompt = "Compare the two images and answer only left or right. " # need to look into how to engineer prompt to include the "key" (surface, color, texture, etc.)
    prompt += attribute['question']

    return prompt

def preprocess_annotations(annotations):
    """
    Make amendments to the annotations prompt to better reflect the image context ("After or Before?" is NOT intuitive at all)
    """
    for annotation in annotations:
        for attribute in annotation['attributes']:
            attribute['question'] = attribute['question'].replace('After or Before', "left or right")

    return annotations

def load_from_splitted_annotations():
    with open(EXISTING_SPLIT_DATASET_PATH, '+r') as file:
        return json.load(file)

def split_test_data(annotations: List, split: float = 0.5):
    """
    We want to carve a number of data (split * len(annotations)) to be only processed.
    
    If that number is not hit, we angrily loop until met_split is `True`
    """

    met_split = False
    carved = []
    max_to_carve = int(len(annotations) * split) # limit number of carved annotations

    while met_split == False:
        for annotation in annotations:
            #  Break the loops once we have hit the maximum number of annotatations to test
            if len(carved) >= max_to_carve:
                met_split = True
                break

            # Carve annotations as we haven't hit the limit yet
            if len(carved) < max_to_carve:
                if random() > 0.5: # Flip a coin if we should include it in the dataset, not related to `split` though
                    carved.append(annotation)
            else:
                break

    # Save the split test data into JSON
    save_split_test_data_json(carved)

    return carved

def save_split_test_data_json(annotations: List):
    with open(EXISTING_SPLIT_DATASET_PATH, "w") as outfile: 
        json.dump(annotations, outfile)


def make_postprocess_directory():
    os.makedirs(POSTPROCESS_PATH, exist_ok=True)

def is_train_or_val_from_name(name: str) -> bool:
    return True if "val/" in name else False

def merge_annotations():
    """
    JSON annotations for both `val/` and `train/` are merged into one big JSON
    """

    merged = []
    attributes_count = 0

    for annotation_source_location in ANNOTATION_SOURCES:
        with open(annotation_source_location, '+r') as file:
            for pair in json.load(file):
                pair['test'] = is_train_or_val_from_name(pair['name'])
                merged.append(pair)
                attributes_count += len(pair['attributes'])

    print(f"There are {attributes_count} attributes.")

    return merged

def scale_images(images: List[PngImagePlugin.PngImageFile], max_dimension: int) -> List[PngImagePlugin.PngImageFile]:
    """
    Resizes an image given a function. It can use any function, but we just use GCF (greatest common factor) here.
    """
    for image in images:
        new_dimensions: tuple[int, int]

        if image.height > image.width:
            new_dimensions = ((image.width * image.height) * max_dimension, max_dimension)
        else:
            new_dimensions = (max_dimension, (image.width * image.height) * max_dimension)

        image.thumbnail(new_dimensions)

    return images

def get_merged_image_dimensions(images: List[PngImagePlugin.PngImageFile], gutter: int = 10) -> tuple[int, int]:
    """
    We need to determine what the new dimensions of merged image is based on the provided images and the defined gutter
    """

    # This assumes we are going to put images side-by-side. Otherwise add logic and code if stacking images.

    new_dimensions = { "width": gutter, "height": 0 }

    # Loop through the images
    for image in images:
        if image.height > new_dimensions['height']:
            new_dimensions['height'] = image.height

        new_dimensions['width'] += image.width

    return (new_dimensions['width'], new_dimensions['height'])

def preprocess_images(annotations):
    """
    Resizes the annotations to the target dimension. Not only that, we also convert the image dataset from PNG to JPG.
    """

    # Loop through the annotations
    for annotation in annotations:
        merged_image_dimensions = { "width": 0, "height": 0 }
        images_to_merge: List[PngImagePlugin.PngImageFile] = []

        # Create the filename of the merged image file
        merged_image_file_name = merged_image_file_name = annotation['contents'][0]['name'] + "__" + annotation['contents'][1]['name'] + TARGET_IMAGE_FILE_EXTENSION

        # Load images
        for image_meta in annotation['contents']:
            image_folder_path = "val/" if annotation['test'] else "train/"

            pair_image_file = Image.open(IMAGES_PATH + "/" + image_folder_path + image_meta['name'] + GUESS_IMAGE_FILE_EXTENSION)
            
            # Define our merged image's dimensions ahead of time
            if pair_image_file.height > merged_image_dimensions['height']:
                merged_image_dimensions['height'] = pair_image_file.height

            merged_image_dimensions['width'] += pair_image_file.width

            images_to_merge.append(pair_image_file)

        if not SKIP_IMAGE_RESIZE:
            # Resize the image
            images_to_merge = scale_images(images_to_merge, SCALE_MAX_DIMENSION)

            # Determine merged image dimensions including gutter after rescaling
            merged_image_dimensions = get_merged_image_dimensions(images_to_merge, MERGED_IMAGE_GUTTER_SIZE)

            # Create new file and add gutter
            merged_image = Image.new('RGB', merged_image_dimensions, (0, 0, 0))

            # Add images
            merged_image.paste(images_to_merge[0], (0, 0)) # first image
            merged_image.paste(images_to_merge[1], (MERGED_IMAGE_GUTTER_SIZE + images_to_merge[0].width, 0))

            # Save file
            merged_image.save(POSTPROCESS_PATH + "/" +  merged_image_file_name)

        # Specify the merged image's filepath for later use on evaluation.
        annotation['filepath'] = POSTPROCESS_PATH + "/" +  merged_image_file_name

    return annotations

def predict(filepath, prompt: str) -> str:
    """
    Predict the given image and the prompt
    """

    # Initialise parameters on the model
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "conv_mode": None,
        "image_file": filepath,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512
    })()

    # FYI: llava\eval\run_llava.py was modified to return the output instead of printing it

    # Evaluate the image and the prompt
    return eval_model(args)

def evaluate_prediction(result: str, answer: str) -> bool:
    """
    Match the dataset's answer (Before/After) to the a given prediction's answer
    """
    if answer == "Before" and result.casefold() == "right":
        return True
    elif answer == "After" and result.casefold() == "left":
        return True
    else:
        return False

def main():
    make_postprocess_directory()

    annotations = merge_annotations()
    annotations = preprocess_annotations(annotations)

    if not SKIP_IMAGE_PREPROCESSING:
        """
        You will want to skip preprocessing the images especially if you have already resized them on a previous test.

        TODO: check if the target image size already matches the source image size and skip the logic altogether. However, do note that the source images are on PNG and still need to be resized to JPG
        """
        annotations = preprocess_images(annotations)

    if USE_EXISTING_SPLIT_DATASET:
        """
        If the dataset has already been split in the respective SAMPLE_DATASET_PERCENT value, let's load that fragment of the dataset
        """
        splitted_annotations = load_from_splitted_annotations()
    else:
        """
        Carve a new dataset based on SAMPLE_DATASET_PERCENT
        """
        splitted_annotations = split_test_data(annotations, SAMPLE_DATASET_PERCENT)

    # Mark start time of predicting the dataset
    start_time = current_milli_time()

    # Start predicting
    predict_results, annotations_size, correct_count, incorrect_count, attributes_size = start_predicting(splitted_annotations)

    save_predict_results(predict_results)

    # Get total time elapsed predicting
    time_elapsed_in_ms = current_milli_time() - start_time

    # Output results of the prediction
    log_info(f"Annotations Size is {annotations_size}")
    log_info(f"Attributes Size is {attributes_size}")
    log_info(f"Correctly predicted {correct_count} attributes")
    log_info(f"Incorrectly predicted {incorrect_count} attributes")
    log_info(f"Time elapsed in milliseconds: {time_elapsed_in_ms}")
    print(f"Annotations Size is {annotations_size}")
    print(f"Attributes Size is {attributes_size}")
    print(f"Correctly predicted {correct_count} attributes")
    print(f"Incorrectly predicted {incorrect_count} attributes")
    print(f"Time elapsed: {time_elapsed_in_ms}")

main()