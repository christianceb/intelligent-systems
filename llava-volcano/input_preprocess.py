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

"""
Options
"""
SCALE_MAX_DIMENSION = 512
SAMPLE_DATASET_PERCENT = 0.005
SKIP_IMAGE_RESIZE = True # Skip the expensive process of resizing

# You will want it that both values below are equal, especially if you are recreating the test dataset.
SKIP_IMAGE_PREPROCESSING = False
USE_EXISTING_SPLIT_DATASET = False

# When not in preprocessing, leave this string to `""`
OVERRIDE_FOLDER = "llava-volcano/"

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
FORCE_RECREATE_MARKING_SET = False
EXISTING_MARKING_KEYS_PATH = OVERRIDE_FOLDER + "marking_keys.json"
EXISTING_PREPROCESSED_ANNOTATIONS_DATASET_PATH = OVERRIDE_FOLDER + "preprocessed_annotations.json"
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


def start_predicting(annotations, marking_set: List[str]):
    """
    Predict the annotation and attributes
    """

    predict_results = []
    annotations_size = len(annotations)
    sample_size = int(annotations_size * SAMPLE_DATASET_PERCENT)
    attributes_size = 0
    correct_count = 0
    incorrect_count = 0

    log_info("About to start predicting", {
        "sample_size": sample_size,
    })

    for index, marking_key in enumerate(marking_set):
        index_annotation = next(i for i, annotation in enumerate(annotations) if annotation['name'] == marking_key)
        annotation = annotations[index_annotation]

        log_info("Running annotation", {
            "index": index_annotation,
            "attribute_name": annotation['name'],
            "percent": int((index / sample_size) * 100),
        })

        for index_attribute, attribute in enumerate(annotation['attributes']):
            attributes_size += 1
            start_time = current_milli_time()
            
            # We need to make our prompt coherent for the model to understand what we want
            prompt = preprocess_prompt(attribute)

            log_info("About to predict attribute in annotation", {
                "annotation_index": index_annotation,
                "attribute": f"{index_attribute}/{len(annotation['attributes'])}",
            })

            merged_image_file_name = create_merged_image_file_name(annotation)
            filepath = POSTPROCESS_PATH + "/" + merged_image_file_name
            
            # Check if the file exists first
            if not os.path.isfile(filepath):
                raise Exception("File to be predicted is not in path.")

            # Have the model make a prediction
            prediction = predict(filepath, prompt)

            log_info("Predicted something", {
                "annotation_name": annotation['name'],
                "annotation_index": index_annotation,
                "attribute": f"{index_attribute}/{len(annotation['attributes'])}",
                "prediction": prediction,
                "expected_prediction": attribute['answer'],
                "translated_expected_prediction": "Right" if attribute['answer'] == "Right" else "Left",
            })

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
                "expected_prediction": attribute['answer'],
                "translated_expected_prediction": "Right" if attribute['answer'] == "Right" else "Left",
                "time_elapsed": current_milli_time() - start_time
            })

        if index >= sample_size:
            break

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

    save_preprocessed_annotations_json(annotations)

    return annotations

def load_from_preprocessed_annotations():
    with open(EXISTING_PREPROCESSED_ANNOTATIONS_DATASET_PATH, '+r') as file:
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
    save_preprocessed_annotations_json(annotations)

    return annotations

def save_preprocessed_annotations_json(annotations: List):
    with open(EXISTING_PREPROCESSED_ANNOTATIONS_DATASET_PATH, "w") as outfile: 
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

def create_merged_image_file_name(annotation) -> str:
    file_name = ""
    
    file_name += annotation['contents'][0]['name']
    file_name += "__"
    file_name += annotation['contents'][1]['name']
    file_name += "__"
    file_name += str(SCALE_MAX_DIMENSION)
    file_name += TARGET_IMAGE_FILE_EXTENSION

    return file_name

def preprocess_images(annotations):
    """
    Resizes the annotations to the target dimension. Not only that, we also convert the image dataset from PNG to JPG.
    """

    start_time = current_milli_time()

    # Loop through the annotations
    for annotation in annotations:
        merged_image_dimensions = { "width": 0, "height": 0 }
        images_to_merge: List[PngImagePlugin.PngImageFile] = []

        # Create the filename of the merged image file
        merged_image_file_name = create_merged_image_file_name(annotation)

        if (not os.path.isfile(POSTPROCESS_PATH + "/" +  merged_image_file_name)):
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

    log_info("Image preprocessing finished", {
        "duration": current_milli_time() - start_time
    })

    return annotations

def predict(filepath, prompt: str) -> str:
    """
    Predict the given image and the prompt
    """

    return "Right" if random() > 0.5 else "Left"

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

def use_marking_set(annotations) -> List[str]:
    marking_set = []

    if os.path.isfile(EXISTING_MARKING_KEYS_PATH) and not FORCE_RECREATE_MARKING_SET:
        with open(EXISTING_MARKING_KEYS_PATH, '+r') as file:
            marking_set = json.load(file)
    else:
        while len(marking_set) < len(annotations):
            for annotation in annotations:
                if len(marking_set) >= len(annotations):
                    break

                if annotation['name'] not in marking_set:
                    if random() > 0.5:
                        marking_set.append(annotation['name'])

        with open(EXISTING_MARKING_KEYS_PATH, "w") as outfile: 
            json.dump(marking_set, outfile)

    return marking_set

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

    marking_set = use_marking_set(annotations)

    # Mark start time of predicting the dataset
    start_time = current_milli_time()

    # Start predicting
    predict_results, annotations_size, correct_count, incorrect_count, attributes_size = start_predicting(annotations, marking_set)

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