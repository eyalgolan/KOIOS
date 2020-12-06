import logging
import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as sig
import json

class SensorData:
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.raw_json = self.get_rawdata_json()
        self.sensor_dataframe = self.get_sensor_data()

    def get_filename(self, in_filename, file_type):
        for fname in os.listdir(self.raw_data_dir):
            if in_filename in fname and file_type in fname:
                return str(fname)

    def get_rawdata_json(self):
        name = self.raw_data_dir

        metis = {"events": self.get_filename("events", "csv"),
                 "sensors": self.get_filename("sensors", "csv")}
        sensors ={"OH1":
                      {"ACC":{"dir":"",
                              "files":[self.get_filename("ACC", "txt")]},
                       "PPG":{"dir":"",
                              "files":[self.get_filename("PPG", "txt")]}},
                  "H10":
                      {"ECG":{"dir":"",
                              "files":[self.get_filename("ECG", "txt")]},
                       "ACC":{"dir":"",
                              "files":[self.get_filename("ACC", "txt")]}}}
        videos = {"phone":
                      {"dir":"phone",
                       "files":[self.get_filename("P", "mp4")]},
                  "face":
                      {"dir":"phone",
                       "files":""},
                  "body":
                      {"dir":"phone",
                       "files":""},
                  "face_gopro":
                      {"dir":"phone",
                       "files":""}}

        data = {"name":name,
                "metis":metis,
                "sensors":sensors,
                "videos":videos}

        raw_json = json.loads(data)
        return raw_json

    def get_sensor_data(self):
        n_sensors = len(self.raw_json["sensors"])

        logging.info(f'Total number of sensors: {n_sensors}')
        files, df = {}, {}
        for sname, sdata in self.raw_json['sensors'].items():
            for dname, ddata in sdata.items():
                sensor = (sname, dname)
                files[sensor] = os.path.join(raw_data_dir, ddata['dir'], ddata['files'][0])
                logging.info(f'Sensor: {sname} Data: {dname} File: {files[sensor]}')
                df[sensor] = mu.get_polar_sensor(files[sensor])
                display(df[sensor].head(5))
        return df