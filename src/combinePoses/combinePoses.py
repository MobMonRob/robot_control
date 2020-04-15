import sys
import zmq
import time
import numpy as np
import json, codecs
import itertools as it
import pandas as pd
import datetime
#import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ergonomicRating import ratingClass

socket = None

def initZMQ():
  global socket
  port = sys.argv[1]

  # Socket to talk to server
  context = zmq.Context()
  socket = context.socket(zmq.PAIR)
  socket.connect("tcp://localhost:%s" % port)

def getConfigAndData():
  # Do Stuff in between
  socket.send_json({"status": "ready"})

  config = socket.recv_json()
  data = socket.recv_json()

  return (config, data)


if __name__ == "__main__":
  #import data
  initZMQ()
  config, data = getConfigAndData()

  data = '[{"LELBW": 90, "RELBW": 95, "LSHOULDER1": 5, "RSHOULDER1": 10},\
          {"LELBW": 110, "RELBW": 120, "LSHOULDER1": 5, "RSHOULDER1": 17}\
          {"LELBW": 70, "RELBW": 70, "LSHOULDER1": 10, "RSHOULDER1": 25},\
          {"LELBW": 80, "RELBW": 85, "LSHOULDER1": 15, "RSHOULDER1": 10}]'
  
  dataStream = json.loads(data)

  print('Data: ', dataStream)
  
  # Calculate Stuff Call Functions...
  time = np.asarray(data['time'])
  data = np.asarray(data['data'])
  print('X: ', time)
  print('y: ', data)

 

  # create valid json
  jsonObject = []
  for i in range(0, len(time)):
    jsonObject.append({'mean': float(data[i]), 'timestamp': float(time[i])})
  print('json: ', jsonObject)
  
  # Send result
  socket.send_json({"status": "result", "result": jsonObject})

  socket.send_json({"status": "finished"})




