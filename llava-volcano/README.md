# Mayon (llava-volcano)

[Mayon](https://en.wikipedia.org/wiki/Mayon) is an implementation of [LLaVa (Large Language and Vision Assistant)](https://llava-vl.github.io/) designed for CSG2341 Intelligent Systems and is meant to be used with the [LLaVa GitHub repository](https://github.com/haotian-liu/LLaVA). Bulk of the work written on this repo is for preprocessing and integrating it with the prediction method of LLaVa. The dataset used here is based off the [EvalAI challenge **Subtle Differences Recognition Challenge - AI Course at ECU**](https://eval.ai/web/challenges/challenge-page/2347/overview).

# Requirements

## Software
- [Conda](https://anaconda.org/anaconda/conda)
- [LLaVa GitHub repository](https://github.com/haotian-liu/LLaVA) if evaluating

## Hardware

If running only the preprocessor, any modern computing device should do depending on how large the input and output dataset size is.

If running the entire evaluation chain, it is **recommended** that you use Amazon Web Services' (AWS) `g5.2xlarge` EC2  instance as it was written in there in SageMaker (not necessary to run exactly in SageMaker). You're free to experiment it with your own hardware, but the recommended VRAM is `24G`, the equivalent of an [NVIDIA RTX 4090](https://www.nvidia.com/en-au/geforce/graphics-cards/40-series/rtx-4090/).

# Usage

## Preprocessor
If running the preprocessor (`input_preprocess.py`) from any machine on a Windows machine:

```powershell
# Create environment
conda create --name llava python=3.12

# List environments to verify (both commands does the same effect)
conda env list
conda info --envs

# Activate environment
conda activate llava

# Install pip packages
pip install -r requirements.txt

# Run the preprocessor
python input_preprocess.py

# Deactivate conda environment
conda deactivate llava
```

## Full evaluation chain
If running the full evaluation chain, you need to clone the [LLaVa GitHub repository](https://github.com/haotian-liu/LLaVA) outside of this repository and copy `evaluate.py` on that folder. You also need to make a minor modification to the clone repository as seen on `evaluate.py`:

```python
# FYI: llava\eval\run_llava.py was modified to return the output instead of printing it
```

Dataset should also ben on the same folder as the cloned LLaVa repository and this repository. Folder should be named `compare_ds` (check `DATASET_PATH`). Refer to tree structure for guidance:
```sh
.
├───llava-volcano
│   └───...
├───llava
│   └───...
├───compare_ds
│   └───images
│       └───train
│           └───...
│       └───val
│           └───...
│   └───annotations
│       └───...
```

`cd` to the cloned repository and run the following:

```bash
# If using AWS SageMaker as it uses `sh` by default
bash

# Create and activate environment
conda create -n llava python=3.10 -y
conda activate llava

# Install packages
pip install --upgrade pip
pip install -e .

# Install additional required package from LLaVa
pip install protobuf

# Run the full evaluation chain
python evaluate.py
```

# How it works
The full evaluation chain follows a sequence of operations:
1. Merges the JSON annotations from both `train/` and `val/` on the provided dataset
2. Pre-process the annotations attributes to make it much more suitable and sensible for the model to process based on our image preprocessing. For example, if the attribute says:
    > The pineapple in Image 1 is rough compared to Image 2. Which is image 1: After or Before?

    then the prompt is changed to:
    > The pineapple in Image 1 is rough compared to Image 2. Which is image 1: left or right?
    
    as the preprocessed image are put on the same image and laid side-by-side.
3. The pair of images (`png`) of each annotation are then used to generate a new compatible image (`jpg`) where the images are resized to the target size (`SCALE_MAX_DIMENSION`) and are put together side-by-side by a gutter(`MERGED_IMAGE_GUTTER_SIZE`).
4. The annotations are split based on the amount of annotations wanted (`SAMPLE_DATASET_PERCENT`). So if it set at `0.5` where there are 10,000 annotations, then a sample dataset of 5,000 annotations are carved.
5. Based on the sampled dataset, we then predict the attributes on the annotation. The prompt is slightly modified (`preprocess_prompt()`) to set expectations on what are the expected answers (left or right)
6. The results are then collated to JSON or CSV and summarised.

# References
- [LLaVa GitHub repository](https://github.com/haotian-liu/LLaVA)
- [LLaVa Homepage](https://llava-vl.github.io/)
- [LLaVA CLI with multiple images](https://github.com/mapluisch/LLaVA-CLI-with-multiple-images?tab=readme-ov-file) (similar implementation)
- [EvalAI challenge](https://eval.ai/web/challenges/challenge-page/2347/overview)