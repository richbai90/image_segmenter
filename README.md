# ImageSegmenter

This Python package provides a class `ImageSegmenter` for annotating images with bounding boxes using various image processing techniques. 

## Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Usage](#usage)
- [Support](#support)
- [Contributing](#contributing)

## Background

Image segmentation and annotation is a crucial part of any machine learning project that deals with images. The `ImageSegmenter` class provided in this package simplifies this process. It uses various image processing techniques like morphological operations, thresholding, and channel selection to help generate accurate bounding boxes. It also includes the ability to visualize the results, save the annotations, and apply various filters.

## Installation

The `ImageSegmenter` package requires:

- OpenCV
- NumPy
- Matplotlib
- VOCWriter (For saving annotations)

To install these dependencies, you can use pip:

```bash
pip install opencv-python numpy matplotlib voc-writer
```

To use the `ImageSegmenter` class, simply include it in your Python script:

```python
from image_segmenter import ImageSegmenter
```

## Usage

Here's an example usage of the `ImageSegmenter` class:

```python
from image_segmenter import ImageSegmenter

# Instantiate the ImageSegmenter
annotator = ImageSegmenter()

# Read an image
image = annotator.read_image('path_to_image')

# Convert to grayscale
image_gray = annotator.select_colorsp(image, 'gray')

# Apply thresholding
thresholded = annotator.threshold(image_gray)

# Apply morphological operations
morphed = annotator.morph_op(thresholded, 'open')

# Find contours
bboxes = annotator.get_bboxes(morphed)

# Visualize the results #
#########################

# draw bounding boxes
annotated = annotator.draw_bboxes(image, bboxes)

# show the image
annotator.display_image(image, annotated)

#########################

# Save the annotations
annotator.save_annotations('path_to_save_annotations', image, bboxes)
```

## TODO
- Write tests
- add multi-label support
- add support for other image formats
- add support for other image processing techniques
- add support for other annotation formats


# Special Thanks
A special thanks to **Kukil**, the author of [this guide](https://learnopencv.com/automated-image-annotation-tool-using-opencv-python/) from which these methods were generated