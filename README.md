# Normal Map Generator

Normal Map Generator is a tool written in Python

## Required

- Python
- Scipy
- Numpy
- Pillow

## Usage
```
python normal_map_generator.py [-h] [-s SMOOTH_VALUE] [-it INTENSITY_VALUE] [-ao AOSTRENGTH_VALUE] input_file
```
Generates the normal and ambient occlusion (AO) map of an image

Maps will be saved alongside the input image, eg. if input is `image.png`
```
.
|- image.png
|- image_AO.png
|- image_Normal.png
```

### Required arguments:

#### input_file
input image path

### Optional arguments:

#### -h, --help
Show help message

#### -s SMOOTH_VALUE, --smooth SMOOTH_VALUE
Smooth gaussian blur applied on the image

#### -it INTENSITY_VALUE, --intensity INTENSITY_VALUE
Intensity of the normal map

#### -ao AOSTRENGTH_VALUE, --aostrength AOSTRENGTH_VALUE
Strength of the ambient occlusion map
