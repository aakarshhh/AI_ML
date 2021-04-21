import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from Environment import *
from Model import *
from tqdm import tqdm
env = CarlaEnv()
a = env.reset()
time.sleep(5)
b =env.step(1)
print(a.shape)
print(b[0].shape)

