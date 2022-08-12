# JPEG [![Profile][title-img]][profile]

[title-img]:https://img.shields.io/badge/-LAVS-blue
[profile]:https://github.com/LAVS-TM

This repository conatins a simple reimplementation of the **JPEG** algorithm.

## Objectives

The main goal was to recode JPEG for color images. In particular, our code is supposed to :
* Manage the color, leaving the user the choice of compression in the **RGB** or **YUV** space, as well as the different **sub-sampling options** (4:4:4, 4:2:2 and 4: 2:0) of chroma.
* Handle the case of images whose dimensions are not necessarily multiples of 8.
* Allow the user to choose the quality index q for the **luminance quantization matrix**.
* Encode via Huffman's JPEG tables the different macro-blocks, to be able to calculate the (true) compression rate of each macro-block (and therefore of the image in total).

## Implementation

The **compression** and the **decompression** process are both implemented in four different steps.

First, the input image is divided into non-overlapping 8 x 8 **macro-blocks**. 

<img src="https://github.com/LAVS-TM/Map-Generation/blob/main/doc/CityExample.png" alt="Example city">


## Usage

A test of our **JPEG** compression algorithm is simply available by executing the `jpeg.py` python file as follow :

```python
python jpeg.py
```
