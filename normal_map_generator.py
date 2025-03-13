import argparse
import math
import numpy as np
from scipy import ndimage
from PIL import Image
import os

def smooth_gaussian(im:np.ndarray, sigma) -> np.ndarray:
    if sigma == 0:
        return im

    im_smooth = im.astype(float)
    kernel_x = np.arange(-3*sigma,3*sigma+1).astype(float)
    kernel_x = np.exp((-(kernel_x**2))/(2*(sigma**2)))

    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis])

    im_smooth = ndimage.convolve(im_smooth, kernel_x[np.newaxis].T)

    return im_smooth

def sobel(im_smooth:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gradient_x = im_smooth.astype(float)
    gradient_y = im_smooth.astype(float)

    kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    gradient_x = ndimage.convolve(gradient_x, kernel)
    gradient_y = ndimage.convolve(gradient_y, kernel.T)

    return gradient_x,gradient_y


def compute_normal_map(gradient_x:np.ndarray, gradient_y:np.ndarray, intensity:float=1.0) -> np.ndarray:
    width = gradient_x.shape[1]
    height = gradient_x.shape[0]
    max_x = np.max(gradient_x)
    max_y = np.max(gradient_y)

    max_value = max_x

    if max_y > max_x:
        max_value = max_y

    normal_map = np.zeros((height, width, 3), dtype=np.float32)

    intensity = 1 / intensity

    strength = max_value / (max_value * intensity)

    normal_map[..., 0] = gradient_x / max_value
    normal_map[..., 1] = gradient_y / max_value
    normal_map[..., 2] = 1 / strength

    norm = np.sqrt(np.power(normal_map[..., 0], 2) + np.power(normal_map[..., 1], 2) + np.power(normal_map[..., 2], 2))

    normal_map[..., 0] /= norm
    normal_map[..., 1] /= norm
    normal_map[..., 2] /= norm

    normal_map *= 0.5
    normal_map += 0.5

    return normal_map


def normalized(im:np.ndarray) -> np.ndarray:
    factor = 1.0/math.sqrt(np.sum(im*im)) # normalize
    return im*factor

def my_gauss(im:np.ndarray) -> np.ndarray:
    return ndimage.uniform_filter(im.astype(float),size=20)

def shadow(im:np.ndarray, strength:float=0.5) -> np.ndarray:
    im1 = im.astype(float)
    im0 = im1.copy()
    im00 = im1.copy()
    im000 = im1.copy()

    for _ in range(0,2):
        im00 = my_gauss(im00)

    for _ in range(0,16):
        im0 = my_gauss(im0)

    for _ in range(0,32):
        im1 = my_gauss(im1)

    im000=normalized(im000)
    im00=normalized(im00)
    im0=normalized(im0)
    im1=normalized(im1)
    im00=normalized(im00)

    shadow=im00*2.0+im000-im1*2.0-im0
    shadow=normalized(shadow)
    mean = np.mean(shadow)
    rmse = np.sqrt(np.mean((shadow-mean)**2))*(1/strength)
    shadow = np.clip(shadow, mean-rmse*2.0,mean+rmse*0.5)

    return shadow

def flip_green(im:np.ndarray) -> None:
    '''
    Invert green channel
    '''
    im[..., 1] = 1.0 - im[..., 1]

def colorize_ao(im:np.ndarray) -> np.ndarray:
    '''
    Map shadow values to greyscale colors
    '''
    blackpoint = 0
    rgb_black = 100
    whitepoint = 180
    rgb_white = 255

    # Convert range to 0-255 uint8
    int_im = np.interp(im, [np.min(im), np.max(im)], [0, 255]).astype(np.uint8)

    # Values < blackpoint to rgb_black, > whitepoint to rgb_white, anything in between linearly interpolated
    colorized = np.interp(int_im, [blackpoint, whitepoint], [rgb_black, rgb_white])

    return np.stack([colorized, colorized, colorized], axis=-1).astype(np.uint8)


def adjust_path(original_path:str, map_type:str) -> str:
    '''
    Path to save the maps alongside the original file eg.
    .
    |- image.png
    |- image_Normal.png
    |- image_AO.png
    '''
    root, ext = os.path.splitext(original_path)
    new_path = f"{root}_{map_type}{ext}"
    print(f"Saving to {new_path}")
    return new_path

def convert(input_file:str, smoothness:float, intensity:float, shadow_strength:float):
    with Image.open(input_file) as f:
        im = np.asarray(f)

    if im.ndim == 3:
        im_grey = np.zeros((im.shape[0],im.shape[1])).astype(float)
        im_grey = (im[...,0] * 0.3 + im[...,1] * 0.6 + im[...,2] * 0.1)
        im = im_grey

    im_smooth = smooth_gaussian(im, smoothness)

    sobel_x, sobel_y = sobel(im_smooth)

    normal_map = compute_normal_map(sobel_x, sobel_y, intensity)
    flip_green(normal_map)

    Image.fromarray((normal_map * 255).astype(np.uint8), mode="RGB").save(adjust_path(input_file,"Normal"))

    im_shadow = shadow(im, shadow_strength)
    im_shadow = colorize_ao(im_shadow)

    Image.fromarray(im_shadow, mode="RGB").save(adjust_path(input_file,"AO"))


def main():
    parser = argparse.ArgumentParser(description='Compute normal and ambient occlusion map of an image')

    parser.add_argument('input_file', type=str, help='input file path')
    parser.add_argument('-s', '--smooth', default=0., type=float, help='smooth gaussian blur applied on the image')
    parser.add_argument('-it', '--intensity', default=1., type=float, help='intensity of the normal map')
    parser.add_argument('-ao', '--aostrength', default=0.5, type=float, help='strength of the AO map')

    args = parser.parse_args()

    input_file = args.input_file
    sigma = args.smooth
    intensity = args.intensity
    shadow_strength = args.aostrength

    convert(input_file, sigma, intensity, shadow_strength)


if __name__ == "__main__":
    main()
