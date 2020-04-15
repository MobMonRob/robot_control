import sys
import zmq
import time
import numpy as np
import json, codecs
import itertools as it
import pandas as pd
import datetime
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ergonomicRating import RatingClass
from optimalTaskPose import main as mainOptimalTaskPose
from distancePlot import main as mainDistancePlot
import xml_rpc_communication as xmlrpc
from grid_interpolating import grid_interpolating

socket = None



if __name__ == "__main__":


  defaultPose = np.array([0,0,0])
  minimalPose = np.array([-10, -10, -10])
  maximalPose = np.array([10, 10, 10])
  workerTarget = np.array([0, -20, -10])
  # mainOptimalTaskPose(defaultPose, minimalPose, maximalPose)

  # mainDistancePlot(defaultPose, minimalPose, maximalPose, workerTarget)
  grid_interpolating(defaultPose, minimalPose, maximalPose, workerTarget)









