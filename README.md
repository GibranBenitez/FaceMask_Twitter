# Twitter Face Image Mining for Recognition of Different Face Mask Types

1. [Intro](#introduction)
2. [Requirements](#requirements)
3. [General Instructions](#face-mask-image-mining)
4. [Preliminary Dataset](#preliminary-dataset)
5. [Face Mask Recognition](#face-mask-recognition)

## Introduction
- This is the repository of our paper **Twitter Face Image Mining for Recognition of Different Face Mask Types**
- Follow the [general instructions](#face-mask-image-mining) to run the complete process for Face Mask Image Mining on Twitter
- If you are interested in the preliminary [dataset download it here (460MB)](https://drive.google.com/file/d/16iVbv-QC_lOuO4fQneTWIeAtkDb1oUTi/view?usp=sharing)
- If you want to try the face mask recognition model only, [follow these instructions](#face-mask-recognition)

## Requirements

* Install the necessary libraries:
  - wget
  - tweepy
  - PyTorch version 1.10.0+ and torchvision 0.3.0+

## Face Mask Image Mining

- Clone this repository `$ git clone https://github.com/GibranBenitez/FaceMask_Twitter.git`
- Update your [Twitter API V2](https://developer.twitter.com/en/docs/twitter-api) Academic Research access level credentials in `./MaskTwitter/credentials.py`
- Download the [ConvNeXt weights (190MB)](https://drive.google.com/file/d/172LJwnRH5Kzop5iWyiuHR_oQtEX9O4F1/view?usp=sharing) on `./MaskClassify/weights/`
- Modify the date range and keywords of faceMaskTwitter.py and demo_covid_classy.py and run these commands. 
```bash
$ python ./RetinaFace/faceMaskTwitter.py
$ python ./MaskClassify/demo_covid_classy.py
```

- Example values for **faceMaskTwitter.py** (keywords and date range):
```python
keywords = ["n95","ffp2","face mask","cubrebocas","barbijo"]
dates = ["2019-12-01","2019-12-31"]
```
- Example values for **demo_covid_classy.py** (keywords and folder name of dates):
```python
keywords = ["n95","ffp2","face mask","cubrebocas","barbijo"]
 dates = "2019-12-01__2022-12-31"
```

## Preliminary Dataset

- We manually annotate 10,500 images based on the four face mask types: surgical masks, cloth masks, respirators, and valved mask, as well as unmasked faces. 
- You can download the dataset from [Google Drive (460MB)](https://drive.google.com/file/d/16iVbv-QC_lOuO4fQneTWIeAtkDb1oUTi/view?usp=sharing) 
- The number of images for train, validation, and test sets are as follows

| Set | Surgical | Cloth | Respirator | Valved | Unmasked | Total |
| --- | -------- | -------- | -------- | -------- | -------- | -------- |
| Train | 2200 | 1700 | 1000 | 1000 | 2100 | 8000 |
| Val. | 200 | 200 | 200 | 100 | 300 | 1000 | 
| Test | 350 | 330 | 300 | 170 | 350 | 1500 | 

## Face Mask Recognition

- We train a ConvNeXt architecture with our preliminary dataset. The results obtained with validation and test sets are **94.8%** and **94.1%**, respectively.
- To run a classification demo you have to download the [ConvNeXt weights (190MB)](https://drive.google.com/file/d/172LJwnRH5Kzop5iWyiuHR_oQtEX9O4F1/view?usp=sharing) on `./MaskClassify/weights/`
- Modify the data path `data_dir`, the output folder `out_dir`, and run `$ python ./MaskClassify/demo_covid_classy.py`
- Example of modifications from **demo_covid_classy.py** (data_dir, includes 10 demo images):
```python
############## Uncomment this section for demo classy only ################    
dates = None
############## Chose pahts for demo classy only ###########################
data_dir = "./MaskClassify/demo_imgs"   
out_dir = "./MaskClassify/output"
```