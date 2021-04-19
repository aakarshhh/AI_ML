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
env = CarEnv()
a = env.reset()
print(a.shape)
