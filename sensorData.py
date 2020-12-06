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

    def get_rawdata_json(self):
        
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