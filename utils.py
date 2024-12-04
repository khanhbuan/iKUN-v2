import os
import math
import json
import random
import logging
import numpy as np
from os.path import join
from collections import defaultdict
import torchvision.transforms.functional as F

import clip
import torch


__all__ = [
    'VIDEOS', 'RESOLUTION', 'WORDS', 'EXPRESSIONS', 'FRAMES',
    'multi_dim_dict', 'expression_conversion', 'SquarePad',
    'save_configs', 'get_logger', 'get_lr', 'set_lr',
    'AverageMeter', 'ProgressMeter', 'tokenize', 'load_from_ckpt', 
    'set_seed', 'WORDS_MAPPING'
]


VIDEOS = {
    'train': [
        '0001', '0002', '0003', '0004', '0006',
        '0007', '0008', '0009', '0010', '0012',
        '0014', '0015', '0016', '0018', '0020',
    ],
    'val': [
        '0005', '0011', '0013'
    ],
    'test': [
        '0005', '0011', '0013'
    ],
}

RESOLUTION = {
    '0001': (375, 1242), '0002': (375, 1242), '0003': (375, 1242), '0004': (375, 1242),
    '0005': (375, 1242), '0006': (375, 1242), '0007': (375, 1242), '0008': (375, 1242),
    '0009': (375, 1242), '0010': (375, 1242), '0011': (375, 1242), '0012': (375, 1242),
    '0013': (375, 1242), '0014': (370, 1224), '0015': (370, 1224), '0016': (370, 1224),
    '0018': (374, 1238), '0020': (376, 1241),
}
WORDS_MAPPING = {
    'auto':             'car',
    'autos':            'car',
    'car':              'car',
    'cars':             'car',
    'automobiles':      'car',
    'vehicles':         'car',
    'vehicle':          'car',
    'motorcars':        'car',
    'motors':           'car',
    'automotives':      'car',
    'vehicular':        'car',
    'population':       'pedestrian',
    'people':           'pedestrian',
    'peoples':          'pedestrian',
    'humans':           'pedestrian',
    'populace':         'pedestrian',
    'persons':          'pedestrian',
    'those':            'pedestrian',
    'pedestrians':      'pedestrian',
    'pedestrian':       'pedestrian',
    'individuals':      'pedestrian',
    'someone':          'pedestrian',
    'folks':            'pedestrian',
    'crowd':            'pedestrian',
    'human':            'pedestrian',
    'walker':           'pedestrian',
    'walkers':          'pedestrian',
    'whose':            'pedestrian',
    'ones':             'pedestrian',
    'passerby':         'pedestrian',
    'passersby':        'pedestrian',
    'citizens':         'pedestrian',
    'standers':         'pedestrian',
    'their':            'pedestrian',
    'commonplace':      'pedestrian',
    'ordinary':         'pedestrian',
    'gentlemen':        'men',
    'males':            'men',
    'male':             'men',
    'men':              'men',
    'guys':             'men',
    'man':              'men',
    'masculine':        'men',
    'dudes':            'men',
    'womem':            "women",
    'lady':             'women',
    'ladies':           'women',
    'women':            'women',
    'females':          'women',
    'female':           'women',
    'womenfolk':        'women',
    'woman':            'women',
    'womens':           'women',
    'fairer':           'women',
}

WORDS = {
    'dropped': [
        'in', 'the', 'with', 'direction', 'of', "currently", "sex", 
        "has", "is", "t", "as", 'ours', 'camera', 'and', 'which', 'are', 
        "side", "from", "each", "other", "be", "can", 'than', 'carrying', 
        'holding', 'a', 'bag', "their", "on", "compared", 'pants', 'who', 
        'horizon', "to", "that", "were", "have", "been", "there"
    ]
}

EXPRESSIONS = {
    'dropped': [
        'women-back-to-the-camera', 'vehicles-which-are-braking', 'men-back-to-the-camera',
        'vehicles-in-horizon-direction', 'cars-which-are-braking', 'cars-in-horizon-direction',
        'males-back-to-the-camera', 'females-back-to-the-camera',
    ],  # these expressions are not evaluated as in TransRMOT
    "0005": [
        "positioned-to-the-left-are-automobiles-that-are-black",
        "the-red-cars-on-the-left",
        "the-cars-before-ours",
        "autos-moving-in-the-opposite-direction-are-black",
        "the-cars-are-traveling-in-the-same-direction-as-ours",
        "left-cars-in-light-color",
        "cars-were-left-in-silver",
        "the-vehicles-to-the-left-are-headed-in-the-same-direction-as-our-car",
        "cars-in-red",
        "black-cars-can-be-seen-on-the-left",
        "in-the-opposite-direction,-cars-in-black",
        "autos-in-red-are-moving-in-the-opposite-direction-as-ours",
        "black-cars",
        "same-direction-autos-on-the-left",
        "cars-in-the-same-direction-of-ours",
        "left-cars-in-silver",
        "left-cars-which-are-parking",
        "left-were-cars-in-light-color",
        "right-autos-in-light-color",
        "autos-that-are-parked-are-red",
        "autos-going-in-the-opposite-direction-on-the-left",
        "left-cars-in-the-opposite-direction-of-ours",
        "the-vehicles-were-going-opposite-to-ours",
        "same-direction-cars-in-black",
        "autos-are-traveling-in-the-opposite-direction-and-are-light-in-color",
        "right-autos-in-silver",
        "autos-on-the-left-are-moving-in-the-same-direction-as-our-vehicle",
        "black-cars-moving-in-the-opposite-direction",
        "black-cars-on-the-left",
        "positioned-on-the-left-are-autos-that-are-white",
        "automobiles-moving-in-the-opposite-direction-of-our-own",
        "cars-that-are-situated-on-the-right",
        "cars-in-the-opposite-direction-of-ours",
        "silver-vehicles-going-in-opposite-directions",
        "silver-vehicles-are-located-to-the-right",
        "there-are-silver-vehicles-on-the-left",
        "opposite-direction-autos-in-silver",
        "motor-vehicles-in-white",
        "the-autos-that-are-parked-are-white",
        "autos-in-the-same-direction-are-black",
        "the-red-autos-that-are-parked",
        "cars-in-motion",
        "opposite-direction-black-cars",
        "autos-in-light-color-moving-in-the-opposite-direction",
        "black-vehicles-moving-in-the-opposite-direction",
        "the-white-cars-were-left",
        "light-color-autos-on-the-left",
        "cars-on-the-right",
        "the-cars-that-are-red-and-parked",
        "the-cars-left-are-the-ones-that-are-currently-parked",
        "opposite-direction-autos-in-light-color",
        "black-opposite-direction-autos",
        "autos-in-the-direction-opposite-of-black",
        "moving-white-cars",
        "silver-cars-are-on-the-left",
        "silver-colored-cars",
        "autos-on-the-left-side-are-moving-in-the-same-direction",
        "the-cars-that-are-parking-have-been-left-behind",
        "vehicles-that-are-white-in-color-are-moving-in-the-same-direction-as-us",
        "automobiles-in-silver-are-on-the-left",
        "cars-of-the-color-black-can-be-seen-on-the-left-side",
        "the-automobiles-in-front-of-ours",
        "light-colored-cars-are-on-the-left",
        "cars-in-black",
        "light-colored-vehicles-moving-in-the-opposite-direction",
        "silver-cars-on-the-right",
        "cars-in-white-color",
        "cars-that-are-white-and-parked",
        "silver-automobiles",
        "vehicles-that-are-white-are-traveling-in-a-direction-opposite-to-ours",
        "the-automobiles-ahead-of-ours",
        "autos-that-are-black-are-situated-on-the-left",
        "red-parked-cars",
        "white-cars-are-in-motion",
        "the-cars-on-the-right-are-silver",
        "opposite-direction-autos-in-black",
        "vehicles-of-the-color-white-are-moving-in-the-opposite-direction-compared-to-the-ego-car",
        "motor-vehicles-that-are-parked",
        "opposite-direction-red-cars",
        "cars-on-our-left-are-driving-in-the-same-direction-as-we-are",
        "the-left-side-has-silver-automobiles",
        "those-red-cars-are-parked",
        "left-red-cars",
        "autos-of-light-color-positioned-on-the-right",
        "silver-autos-on-the-right",
        "right-cars-in-silver",
        "vehicles-in-light-hues-traveling-in-the-opposite-direction",
        "to-the-left-are-red-cars",
        "cars-in-silver",
        "red-colored-vehicles",
        "red-parked-autos",
        "cars-in-light-color",
        "automobiles-are-located-to-the-left",
        "light-colored-cars",
        "black-cars-heading-in-the-same-direction",
        "autos-of-white-color-were-left",
        "cars-which-are-parked",
        "autos-in-light-color-were-left",
        "autos-in-a-light-color-are-located-on-the-right",
        "light-colored-cars-are-going-in-the-opposite-direction",
        "vehicles-in-white-are-driving-in-the-opposite-direction-from-the-ego-car",
        "these-white-autos-are-parked",
        "cars-that-are-white-and-moving",
        "positioned-on-the-left-are-cars-that-are-silver",
        "vehicles-moving-in-the-opposite-direction-are-of-a-light-color",
        "cars-in-white",
        "right-cars-in-light-color",
        "the-cars-are-to-the-left",
        "cars-on-the-left",
        "automobiles-which-are-white-are-going-in-the-same-direction-as-our-vehicle",
        "left-cars-in-the-same-direction-of-ours",
        "vehicles-that-are-light-colored-are-on-the-right-side",
        "vehicles-that-are-white-are-moving-in-the-opposite-direction-that-we-are-driving",
        "white-cars-are-moving-in-the-direction-of-the-ego-car",
        "autos-that-are-on-the-left-side-are-going-in-the-same-direction",
        "light-color-cars-were-left",
        "left-cars-in-black",
        "the-white-cars-are-following-the-same-direction-as-the-ego-car",
        "cars-coming-from-the-opposite-direction-as-ours",
        "autos-located-to-the-right",
        "light-color-cars-on-the-right",
        "vehicles-that-are-red-are-traveling-in-the-opposite-direction-of-the-car-with-the-ego",
        "black-cars-moving-in-the-same-direction",
        "autos-that-are-red-are-located-on-the-left",
        "silver-cars",
        "the-automobiles-are-moving-opposite-to-the-way-our-vehicle-is-traveling",
        "the-autos,-which-are-white,-are-parked",
        "silver-cars-can-be-seen-on-the-right",
        "there-are-black-cars-positioned-on-the-left",
        "automobiles-in-white-are-headed-in-the-opposite-direction-from-the-ego-car",
        "the-cars-are-heading-in-the-direction-of-our-car",
        "white-autos",
        "silver-autos-on-the-left",
        "automobiles-moving-in-the-opposite-direction-on-the-left",
        "vehicles-opposite-to-ours-were-left-there",
        "cars-that-are-red-are-traveling-in-the-opposite-direction-compared-to-our-own",
        "to-the-right-are-autos-in-light-colors",
        "autos-in-silver-moving-in-the-opposite-direction",
        "opposite-direction-autos-that-are-black",
        "automobiles-in-light-hues-can-be-found-on-the-left",
        "autos-in-the-opposite-direction-are-red",
        "the-left-side-has-cars-in-light-colors",
        "parked-white-cars",
        "vehicles-of-a-light-hue-driving-in-opposite-directions",
        "the-black-cars-were-left",
        "red-autos-that-are-parked",
        "black-same-direction-cars",
        "silver-cars-are-situated-on-the-left",
        "vehicles-driving-in-the-same-direction-as-ours",
        "automobiles-in-transit",
        "vehicles-painted-red-on-the-left-side",
        "red-painted-cars",
        "on-the-right,-silver-cars-are-located",
        "red-autos-which-are-parked",
        "moving-autos",
        "opposite-direction-autos-on-the-left",
        "red-autos-going-in-the-opposite-direction-of-the-ego-car",
        "vehicles-that-are-red-and-moving-in-the-opposite-direction-of-the-ego-car",
        "cars-moving-in-the-opposite-direction-are-red",
        "red-automobiles-are-moving-in-the-opposite-direction-of-the-car-being-driven-by-the-ego",
        "the-vehicles-before-ours",
        "autos-in-black-were-abandoned",
        "silver-cars-are-going-in-the-opposite-direction",
        "same-direction-cars-on-the-left",
        "white-cars-are-going-in-the-opposite-direction-from-where-we-are-heading",
        "opposite-direction-autos-in-red",
        "red-autos-going-in-the-opposite-direction-of-ours",
        "cars-in-front-of-ours",
        "parked-autos-are-white",
        "vehicles-going-in-the-opposite-direction-are-on-the-left",
        "vehicles-are-positioned-on-the-left",
        "autos-facing-the-opposite-way-are-black",
        "autos-in-red-moving-in-the-opposite-direction",
        "vehicles-painted-in-red",
        "silver-cars-on-the-left",
        "there-are-cars-parked-in-the-same-direction-as-our-car-on-the-left-side",
        "red-autos-are-going-in-the-opposite-direction",
        "moving-cars",
        "cars-on-the-left-are-traveling-in-the-opposite-direction",
        "parked-red-cars",
        "moving-cars-are-white",
        "white-automobiles-are-headed-in-the-opposite-direction-of-the-ego-car",
        "cars-moving-in-the-same-direction-are-black",
        "the-vehicles-ahead-of-ours",
        "autos-that-are-light-in-color",
        "same-direction-autos-in-black",
        "automobiles-positioned-on-the-right",
        "red-autos-on-the-left",
        "the-cars-in-red-are-traveling-in-the-opposite-direction",
        "autos-going-in-the-same-direction-are-located-on-the-left-side",
        "those-on-the-right-are-silver-autos",
        "cars-that-are-stationary",
        "parked-autos",
        "automobiles-in-a-light-shade-going-in-opposite-paths",
        "cars-positioned-to-the-left-are-silver",
        "those-automobiles-turned-in-the-opposite-direction-of-ours",
        "the-vehicles-are-driving-in-the-opposite-direction-from-the-one-we-are-going"
    ],
    "0011": [
        "cars-that-were-silver-left",
        "vehicles-driving-in-the-same-direction-as-ours",
        "blue-cars-which-are-moving",
        "autos-of-a-light-color-are-located-on-the-left",
        "swift-red-autos",
        "autos-that-are-red-on-the-left-side",
        "cars-in-motion-to-the-left",
        "the-automobiles-are-white",
        "cars-that-are-silver",
        "autos-in-front-of-ours",
        "automobiles-moving-in-the-contrary-direction-to-ours",
        "those-who-are-walking-are-pedestrians",
        "the-pedestrians-has-a-yellow-t-shirt",
        "cars-that-applied-brakes",
        "turning-cars",
        "cars-in-the-opposite-direction-of-ours",
        "right-cars-in-white",
        "cars-with-light-colors-are-situated-on-the-right",
        "parking-cars",
        "moving-cars-are-of-light-color",
        "opposite-direction-autos-on-the-left",
        "faster-autos-in-red",
        "light-colored-cars-on-the-right",
        "light-colored-cars-are-moving",
        "the-cars-are-white",
        "silver-automobiles-were-abandoned-on-the-left-side",
        "the-vehicles-in-front-of-ours",
        "positioned-on-the-right-are-black-cars",
        "transferring-vehicles",
        "the-automobiles-colored-red-that-are-leading-ours",
        "autos-in-black",
        "the-cars-that-are-quicker-than-ours",
        "autos-which-are-faster-than-ours",
        "autos-on-the-right",
        "autos-of-a-light-color-in-motion",
        "the-cars-on-the-left-are-parking",
        "autos-in-the-same-direction-of-ours",
        "the-automobiles-that-are-blue-are-in-motion",
        "automobiles-that-are-faster-than-the-self-centered-car-and-are-painted-red",
        "the-cars-are-moving-in-the-same-direction-as-ours",
        "the-car-with-an-ego-has-red-cars-in-front-of-it",
        "the-cars-which-are-faster-than-our-own-and-are-red",
        "the-cars-in-front-of-our-car",
        "silver-moving-autos",
        "automobiles-situated-on-the-left",
        "positioned-on-the-left-are-cars-that-are-light-in-color",
        "the-cars-were-left-in-black",
        "black-automobiles-on-the-left-side",
        "cars-on-the-right",
        "on-the-right-side,-there-are-white-cars",
        "cars-that-are-black-are-situated-on-the-right-side",
        "those-cars-which-are-parked-are-on-the-left",
        "autos-in-front-of-the-ego-car",
        "those-who-wear-t-shirts",
        "motor-vehicles-traveling-in-the-opposite-direction-from-us",
        "individuals-on-foot-standing",
        "the-cars-that-outpace-ours",
        "the-moving-autos-are-of-a-light-color",
        "cars-right-in-front",
        "right-autos-in-white",
        "folks",
        "the-cars-that-are-parked-on-the-right",
        "automobiles-directly-in-front",
        "the-white-cars-on-the-left",
        "those-with-a-yellow-t-shirt",
        "moving-cars-towards-the-left",
        "cars-that-slowed-down-using-the-brakes",
        "positioned-on-the-left-are-cars-that-are-white",
        "the-cars-that-are-faster-than-our-own-and-are-red",
        "the-parked-cars-are-on-the-left",
        "human-beings",
        "on-the-right-side-are-cars-with-light-colors",
        "those-red-cars-positioned-in-front-of-ours",
        "blue-vehicles-that-are-in-motion",
        "cars-of-a-light-color-are-located-on-the-left",
        "vehicles,-black-in-color,-positioned-to-the-left",
        "pedestrians-in-motion",
        "moving-automobiles-with-silver-hue",
        "vehicles-are-positioned-to-the-right",
        "autos-made-of-silver-were-left-behind",
        "there-are-cars-on-the-right",
        "the-vehicles-in-motion-are-blue",
        "on-the-right-side,-there-are-light-colored-automobiles",
        "the-cars-were-left-with-a-white-paint-job",
        "cars-in-silver",
        "pedestrians-in-yellow-t-shirts-walking",
        "vehicles-directly-in-front-of-you",
        "the-cars-on-the-left-are-black",
        "rapid-red-cars",
        "cars-moving-parallel-to-our-direction",
        "humans",
        "on-the-right-are-black-cars",
        "vehicles-are-being-parked",
        "blue-cars-in-motion",
        "folks-in-t-shirts",
        "blue-cars-that-are-moving",
        "moving-pedestrians",
        "pedestrians",
        "parking-autos",
        "cars-positioned-to-the-left",
        "cars-that-were-silver-and-to-the-left",
        "autos-of-black-color-are-on-the-left",
        "cars-that-outpace-ours",
        "faster-red-cars-than-ours",
        "the-automobiles-are-heading-in-the-same-direction-as-ours",
        "cars-are-moving-that-are-colored-red",
        "the-autos,-which-are-parked,-are-on-the-right",
        "relocating-automobiles",
        "the-automobiles-in-front-of-ours",
        "moving-autos-of-a-red-color",
        "the-left-side-features-automobiles-in-light-shades",
        "the-autos-that-are-parking-are-on-the-right",
        "cars-moving-have-a-light-color",
        "the-autos-moving-are-a-light-color",
        "automobiles-that-used-their-brakes-to-stop",
        "the-light-colored-autos-are-moving",
        "the-vehicles-that-are-faster-than-ours",
        "pedestrians-standing",
        "light-colored-cars-were-placed-on-the-left",
        "the-cars-on-our-left-are-traveling-in-the-opposite-direction",
        "faster-than-the-ego-car,-red-cars-speed-ahead",
        "pedestrians-on-foot",
        "cars-that-are-traveling",
        "moving-vehicles-of-a-light-color",
        "speedier-automobiles-in-a-red-hue",
        "folks-with-a-bicycle",
        "to-the-left-are-light-colored-vehicles",
        "right-autos-in-black",
        "black-automobiles-are-on-the-left",
        "someone-on-foot-is-sporting-a-t-shirt",
        "the-red-cars-are-slowing-down",
        "pedestrians-who-are-walking",
        "walkers-who-are-pedestrians",
        "cars-in-the-same-direction-of-ours",
        "the-autos-that-are-currently-parked-are-on-the-right",
        "there-are-cars-colored-black",
        "the-red-cars-are-faster-than-the-ego-car",
        "standing-persons",
        "autos-in-white-are-the-right-ones",
        "cars-with-light-colors-are-moving",
        "silver-automobiles",
        "blue-moving-autos",
        "individuals-on-foot-with-a-bicycle",
        "autos-in-light-color-are-left",
        "moving-cars",
        "silver-moving-cars",
        "cars-that-exceed-the-speed-of-our-own",
        "the-autos-of-a-light-color-are-moving",
        "automobiles-to-the-left",
        "silver-cars-that-were-on-the-left",
        "vehicles-changing-direction",
        "the-cars-in-red-ahead-of-ours",
        "the-right-side-features-black-automobiles",
        "moving-autos-are-blue",
        "those-individuals-walking",
        "the-red-vehicles-that-are-in-front-of-ours",
        "automobiles-heading-in-the-opposite-direction-of-our-vehicles",
        "the-red-vehicles-that-are-decelerating",
        "abandoned-autos-were-white",
        "right-autos-in-light-color",
        "faster-cars-in-red",
        "standing-pedestrians",
        "rapid-vehicles-colored-red",
        "the-yellow-t-shirt-belongs-to-that-persons",
        "opposite-direction-vehicles-are-to-the-left",
        "cars-that-are-black-are-on-the-right",
        "moving-cars-painted-in-black",
        "autos-that-are-on-the-left-side-are-white",
        "faster-red-cars-than-the-ones-we-have",
        "silver-automobiles-are-on-the-move",
        "our-left-hand-side-has-cars-going-in-the-opposite-direction",
        "cars-making-turns",
        "cars-that-are-black-can-be-found-on-the-left",
        "the-autos-are-in-front-of-the-car-that-has-an-ego",
        "cars-in-motion-that-are-black",
        "autos-on-the-left",
        "the-automobiles-in-front-of-our-car",
        "those-individuals",
        "automobiles-on-the-right-side",
        "cars-that-are-black-and-on-the-left",
        "the-red-automobiles-are-faster-than-our-car",
        "cars-are-being-parked",
        "silver-vehicles-were-left-on-the-left",
        "cars-that-are-blue-and-moving",
        "light-color-autos-on-the-left",
        "cars-that-came-to-a-stop",
        "faster-than-the-ego-car-are-red-cars",
        "those-people-with-yellow-t-shirts",
        "autos-in-white",
        "pedestrians-walking",
        "cars-in-white",
        "moving-vehicles-are-silver",
        "standing-people",
        "the-cars-that-are-parked-are-on-the-left",
        "cars-painted-in-light-shades",
        "right-cars-which-are-parking",
        "autos-that-are-red-and-moving",
        "autos-painted-blue-are-situated-on-the-left-side",
        "vehicles-directly-in-front",
        "turning-autos",
        "those-with-yellow-t-shirts",
        "automobiles-that-are-blue-and-moving",
        "autos-turning",
        "those-faster-red-cars-outpace-ours",
        "silver-cars",
        "the-moving-cars-are-blue",
        "the-cars-are-positioned-ahead-of-the-ego-car",
        "the-red-cars-outpace-the-ego-car",
        "the-red-autos-that-are-stopping",
        "automobiles-headed-in-the-reverse-direction-of-ours",
        "blue-moving-cars",
        "the-red-automobiles-are-quicker-than-the-ego-vehicle",
        "opposite-direction-cars-on-the-left",
        "light-hued-cars-that-are-currently-moving",
        "light-color-autos-on-the-right",
        "cars-moving-have-light-colors",
        "cars-in-movement-with-a-light-color",
        "the-white-cars-were-left",
        "light-colored-cars-are-on-the-left",
        "light-colored-vehicles-were-left-behind",
        "t-shirt-wearing-individuals",
        "cars-traveling-in-the-opposite-direction-of-our-vehicles",
        "silver-cars-left",
        "blue-cars-that-are-in-motion",
        "cars-that-are-blue-and-are-on-the-left-side",
        "automobiles-parked",
        "moving-autos-that-are-light-colored",
        "autos-left-were-in-white"
    ],
    "0013": [
        "people-whose-t-shirt-is-white",
        "the-pedestrians-are-wearing-a-white-t-shirt-and-are-accompanied-by-a-bicycle",
        "men-wearing-white-t-shirt",
        "those-whose-t-shirt-is-white",
        "males-are-situated-to-the-right",
        "those-located-to-the-left",
        "males-dressed-in-white-t-shirts",
        "the-pedestrians-are-wearing-a-green-t-shirt",
        "people-in-white-t-shirts",
        "men-who-walk",
        "someone-is-standing-on-the-left",
        "women-back-to-the-camera",
        "guys-stand-on-the-right-side",
        "pedestrians-on-the-left",
        "females",
        "positioned-on-the-left-are-vehicles",
        "those-women-who-are-wearing-a-white-t-shirt",
        "members-of-the-fairer-sex",
        "people-donning-green-shirts",
        "the-pedestrians-shirt-is-white",
        "right-parked-cars-in-black",
        "whose-t-shirt-is-white",
        "men-wearing-a-white-t-shirt",
        "right-people-who-are-walking",
        "the-men-are-facing-in-the-opposite-direction-of-the-camera",
        "pedestrians-wearing-green-t-shirt",
        "ladies",
        "the-individuals-on-the-right-are-standing",
        "folks-positioned-to-the-right",
        "those-walking-are-the-right-people",
        "those-wearing-a-green-t-shirt",
        "persons-wearing-green-colored-tees",
        "standing-women",
        "people-on-the-right",
        "people-to-the-right",
        "right-side-black-cars",
        "those-walking-were-left",
        "on-the-right-are-men",
        "feminine-individuals-on-the-right-side",
        "the-individuals-walking-are-situated-to-the-left",
        "automobiles-positioned-on-the-right-hand-side",
        "moving-right-pedestrians",
        "males-are-on-the-left-side",
        "the-ladies-that-are-on-the-left-side",
        "men-are-situated-on-the-left",
        "the-green-t-shirt-of-the-persons",
        "walking-males",
        "females-back-to-the-camera",
        "women-with-a-white-t-shirt",
        "women-holding-a-bag",
        "walkers-were-left-behind",
        "the-green-t-shirt-belongs-to-the-pedestrians",
        "females-located-on-the-right-side",
        "the-people-are-wearing-a-green-t-shirt",
        "people-situated-on-the-left",
        "people-who-have-green-t-shirts",
        "men-wearing-green-t-shirt",
        "walkers-were-left",
        "the-pedestrians-are-wearing-a-white-t-shirt",
        "standing-females",
        "males-on-the-left",
        "those-standing-on-the-right",
        "autos-are-positioned-on-the-right",
        "autos-were-left-parked-in-silver",
        "the-men-are-situated-on-the-right",
        "those-to-the-left",
        "persons-who-are-standing-are-on-the-right",
        "women",
        "the-people-own-a-green-t-shirt",
        "autos-in-black-parked-on-the-right",
        "people-are-standing-on-the-right-side",
        "pedestrians-on-the-right",
        "a-green-t-shirt-is-being-worn-by-the-pedestrians",
        "women-with-a-bike",
        "the-individuals-are-wearing-a-white-shirt",
        "gentlemen",
        "the-men-are-wearing-a-green-t-shirt",
        "pedestrians-positioned-to-the-right",
        "persons-who-are-walking-were-left",
        "female-with-a-bag",
        "men-wearing-a-green-t-shirt",
        "pedestrians-wearing-white-t-shirt",
        "rightward-pedestrians-movement",
        "men-wearing-white-t-shirts",
        "males-on-the-right",
        "men-back-to-the-camera",
        "females-that-are-upright",
        "standing-people-on-the-right",
        "people-who-have-white-t-shirts",
        "right-cars-in-black",
        "the-pedestrians-t-shirt-is-white",
        "standing-persons-on-the-right",
        "people-whose-t-shirt-is-green",
        "the-people-who-are-standing-are-positioned-on-the-left",
        "the-pedestrians-who-are-walking-on-the-right",
        "those-on-the-right-are-men",
        "guys-sporting-a-green-tee",
        "women-located-on-the-right-side",
        "those-walking",
        "standing-pedestrians-on-the-right",
        "walker-with-a-bicycle",
        "persons-whose-t-shirt-is-white",
        "people-wearing-green-t-shirt",
        "pedestrians-located-on-the-right",
        "men-walking",
        "individuals-clothed-in-white-t-shirts",
        "those-dressed-in-white-t-shirts",
        "right-persons-who-are-walking",
        "women-carrying-a-bag",
        "standing-pedestrians-on-the-left",
        "walking-females",
        "the-females-are-located-to-the-left",
        "guys-wearing-white-t-shirts",
        "women-on-the-right",
        "men",
        "the-pedestrians-who-are-walking-on-the-left",
        "those-who-are-females-in-white-t-shirt",
        "women-situated-on-the-right-side",
        "in-silver,-cars-were-parked",
        "individuals-who-have-a-white-t-shirt",
        "pedestrians-whose-t-shirt-is-white",
        "men-who-have-white-t-shirts-on",
        "males",
        "those-on-the-left-side",
        "the-pedestrians-on-the-right-who-are-walking",
        "persons-wearing-white-t-shirt",
        "persons-on-the-left",
        "standing-men",
        "the-parked-cars-to-the-right-are-black",
        "the-individuals-walking-are-dressed-in-a-green-t-shirt",
        "silver-vehicles-were-left-parked",
        "those-in-possession-of-a-green-t-shirt",
        "the-females-are-oriented-in-a-way-that-their-backs-are-visible-to-the-camera",
        "males-standing",
        "individuals-in-green-t-shirts",
        "females-on-the-left",
        "right-parked-autos-in-black",
        "pedestrians-with-a-bike",
        "there-are-men-situated-on-the-left",
        "white-t-shirt-wearing-females",
        "walking-men",
        "those-in-green-tops",
        "individuals-are-standing-on-the-right",
        "the-individuals-who-are-walking-are-right-persons",
        "women-in-a-white-t-shirt",
        "people-wearing-white-t-shirt",
        "womenfolk",
        "persons-on-the-right-side",
        "men-are-in-possession-of-a-green-t-shirt",
        "those-with-white-t-shirts",
        "on-the-left-are-females",
        "the-females-t-shirt-is-white",
        "cars-were-parked-with-a-silver-hue",
        "people-sporting-a-green-t-shirt",
        "guys",
        "right-autos-in-black",
        "those-women-that-are-on-the-right",
        "women-with-their-backs-facing-the-camera",
        "right-pedestrians-who-are-walking",
        "white-t-shirt-wearing-men",
        "males-with-a-green-top",
        "women-with-a-white-shirt",
        "men-whose-t-shirt-is-green",
        "women-standing-up",
        "the-color-of-the-t-shirt-worn-by-the-women-is-white",
        "the-attire-of-the-men-is-a-green-t-shirt",
        "standing-people-on-the-left",
        "males-with-a-green-t-shirt",
        "silver-cars-were-left-parked",
        "those-of-the-female-gender-are-positioned-towards-the-left",
        "cars-are-positioned-to-the-left",
        "people-with-a-white-t-shirt",
        "automobiles-on-the-left-hand-side",
        "individuals-wearing-white-t-shirts",
        "the-pedestrians-who-are-on-the-right-and-walking",
        "automobiles-are-located-on-the-right",
        "green-t-shirt-wearing-men",
        "on-the-left,-there-are-silver-autos",
        "silver-parked-autos-were-left",
        "the-people-walking-are-the-right-ones",
        "males-back-to-the-camera",
        "the-cars-parked-on-the-right-are-black",
        "the-fairer-sex",
        "women-dressed-in-a-white-t-shirt",
        "females-in-a-white-t-shirt",
        "standing-males",
        "cars-were-parked-in-silver",
        "persons-wearing-green-t-shirt",
        "the-pedestrians-has-a-green-t-shirt",
        "standing-persons-on-the-left",
        "cars-positioned-to-the-left",
        "the-color-of-the-t-shirt-worn-by-the-pedestrians-is-white",
        "silver-autos-were-parked-and-left",
        "pedestrians-whose-t-shirt-is-green",
        "males-with-their-backs-to-the-camera",
        "those-wearing-green-t-shirts",
        "women-on-the-left",
        "persons-on-the-right",
        "men-are-positioned-to-the-left",
        "persons-whose-t-shirt-is-green",
        "the-individuals-with-the-bike-are-wearing-a-white-t-shirt",
        "parked-cars-that-are-silver-were-left-behind",
        "people-on-the-left",
        "women-facing-away-from-the-camera",
        "walking-women",
        "the-green-t-shirt-is-worn-by-the-people",
        "females-on-the-right"
    ]
}


FRAMES = {
    '0005': (0, 296),
    '0011': (0, 372),
    '0013': (0, 339),
}  # 视频起止帧

def set_seed(seed):
    """
    ref: https://blog.csdn.net/weixin_44791964/article/details/131622957
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def multi_dim_dict(n, types):
    if n == 0:
        return types()
    else:
        return defaultdict(lambda: multi_dim_dict(n-1, types))


def expression_conversion(expression):
    """expression => expression_new"""
    expression = expression.replace("(1)", "").replace(",", "").replace('-', ' ')
    words = expression.split(' ')
    expression_converted = ''
    for word in words:
        if word in WORDS['dropped']:
            continue
        if word in WORDS_MAPPING:
            word = WORDS_MAPPING[word]
        expression_converted += f'{word} '
    expression_converted = expression_converted[:-1]
    return expression_converted


class SquarePad:
    """Reference:
    https://discuss.pytorch.org/t/how-to-resize-and-pad-in-a-torchvision-transforms-compose/71850/10
    """
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


def save_configs(opt):
    configs = vars(opt)
    os.makedirs(opt.save_dir, exist_ok=True)
    json.dump(
        configs,
        open(join(opt.save_dir, 'config.json'), 'w'),
        indent=2
    )


def get_logger(save_dir):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filename = join(save_dir, 'log.txt')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][%(levelname)s] %(message)s')
    # writting to file
    file_handler = logging.FileHandler(filename, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # display in terminal
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)
    return logger


def get_lr(opt, curr_epoch):
    if curr_epoch < opt.warmup_epoch:
        return (
            opt.warmup_start_lr
            + (opt.base_lr - opt.warmup_start_lr)
            * curr_epoch
            / opt.warmup_epoch
        )
    else:
        return (
            opt.cosine_end_lr
            + (opt.base_lr - opt.cosine_end_lr)
            * (
                math.cos(
                    math.pi * (curr_epoch - opt.warmup_epoch) / (opt.max_epoch - opt.warmup_epoch)
                )
                + 1.0
            )
            * 0.5
        )


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, prefix="", lr=0.):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches, lr)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    @staticmethod
    def _get_batch_fmtstr(num_batches, lr):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + '] [lr: {:.2e}]'.format(lr)


def tokenize(text):
    token = clip.tokenize(text)
    return token


def load_from_ckpt(model, ckpt_path, model_name='model'):
    print(f'load from {ckpt_path}...')
    ckpt = torch.load(ckpt_path)
    epoch = ckpt['epoch']
    model.load_state_dict(ckpt[model_name])
    return model, epoch