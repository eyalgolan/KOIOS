import logging
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import scipy.signal as sig
from datetime import timedelta

class SensorData:
    """
    A class responsible for collecting the sensor data from the files
    """
    def __init__(self, raw_data_dir):
        self.raw_data_dir = raw_data_dir
        self.raw_json = self.get_rawdata_json()
        self.sensor_dataframe = self.create_dataframe_of_sensors()
        self.sdf = self.create_sensors_dataframe()
        self.pdf = self.handle_sdf_cleanup()
        self.find_common_period()

    def get_video_filename(self):
        """

        :return:
        """
        data_dict = json.loads(self.raw_json)
        video_filename = data_dict["videos"]["phone"]["files"][0]
        return video_filename
    def get_filename(self, in_filename, file_type):
        """

        :param in_filename:
        :param file_type:
        :return:
        """
        for fname in os.listdir(self.raw_data_dir):
            if in_filename in fname and file_type in fname:
                return str(fname)

    def get_rawdata_json(self):
        """

        :return:
        """
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

        raw_json = json.dumps(data, indent=4)
        with open("data_file.json", "w") as write_file:
            json.dump(data, write_file, indent=4)
        return raw_json

    def get_polar_sensor(self, sensor_file, sensor_cols=None):
        """

        :param sensor_file:
        :param sensor_cols:
        :return:
        """
        if sensor_cols is None:
            df = pd.read_csv(sensor_file, delimiter=" ")
        else:
            df = pd.read_csv(sensor_file, delimiter=" ", names=sensor_cols,
                             skiprows=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns').dt.round(
            '1ms')
        df['timestamp'] = df['timestamp'] + pd.DateOffset(years=30, days=-1)
        df = df.set_index("timestamp")
        return df

    def create_dataframe_of_sensors(self):
        """

        :return:
        """
        logging.info("Obtaining collected data ...")
        data_dict = json.loads(self.raw_json)
        n_sensors = len(data_dict["sensors"])

        logging.info(f'Total number of sensors: {n_sensors}')
        files, df = {}, {}
        for sname, sdata in data_dict['sensors'].items():
            for dname, ddata in sdata.items():
                sensor = (sname, dname)
                files[sensor] = os.path.join(self.raw_data_dir, ddata['dir'], ddata['files'][0])
                logging.info(f'Sensor: {sname} Data: {dname} File: {files[sensor]}')
                df[sensor] = self.get_polar_sensor(files[sensor])
                #display(df[sensor].head(5))
        return df

    def create_sensors_dataframe(self):
        sensor_cols = ["Time", "Junk1", "Sensor", "X", "Y", "Z", "UX", "UY", "UZ",
                       "Junk2"]
        data_dict = json.loads(self.raw_json)
        sensor_files = data_dict['metis']['sensors']

        if isinstance(sensor_files, list):
            frames = []
            for i, f in enumerate(sensor_files):
                frames.append(
                    pd.read_csv(os.path.join(self.raw_data_dir, f), skiprows=1,
                                names=sensor_cols, engine="python",
                                usecols=[i for i in range(10)]))
                sdf = pd.concat(frames, ignore_index=True)
        else:
            sdf = pd.read_csv(os.path.join(self.raw_data_dir, sensor_files), skiprows=1,
                              names=sensor_cols, engine="python",
                              usecols=[i for i in range(10)])

        sdf['DT'] = pd.to_datetime(sdf.Time, errors='coerce',
                                   format='%Y-%m-%d_%H:%M:%S:%f')
        # sdf['DT'] = pd.to_datetime(sdf.Time, errors='coerce', format='%Y-%m-%d %H:%M:%S:%f')
        sdf = sdf.drop(labels=["Junk1", "Junk2", "Time"], axis=1)
        sdf['Sensor'] = sdf['Sensor'].astype('category')

        sdf.to_csv("sdf.csv")
        return sdf

    def handle_sdf_cleanup(self):
        idx = pd.date_range(start=self.sdf.iloc[0]['DT'], end=self.sdf.iloc[-1]['DT'],
                            freq='.01S')

        logging.info('Cleaning accelerometer')
        acc = self.clean_sensor(self.sdf.loc[
                               self.sdf["Sensor"] == "ICM42605M Accelerometer", [
                                   'DT',
                                   'X',
                                   'Y',
                                   'Z']],
                           idx)
        logging.info('Cleaning magnetometer')
        mag = self.clean_sensor(self.sdf.loc[
                               self.sdf["Sensor"] == "AK09918 Magnetometer", ['DT',
                                                                         'X',
                                                                         'Y',
                                                                         'Z']],
                           idx)
        mag = mag.reindex(index=acc.index, method='nearest')
        logging.info('Cleaning gyroscope')
        gyr = self.clean_sensor(
            self.sdf.loc[
                self.sdf["Sensor"] == "ICM42605M Gyroscope", ['DT', 'X', 'Y', 'Z']],
            idx)
        gyr = gyr.reindex(index=acc.index, method='nearest')

        self.sdf = self.sdf.set_index('DT')

        sensors = ['acc', 'gyr', 'mag']
        axes = ['x', 'y', 'z']
        pdf = pd.concat([acc, gyr, mag], axis=1)
        pdf.columns = pd.MultiIndex.from_product([sensors, axes],
                                                 names=['Sensor', 'Axis'])
        pdf.to_csv("pdf.csv")
        sensors = ['acc', 'gyr', 'mag']
        axes = ['x', 'y', 'z']
        pdf = pd.concat([acc, gyr, mag], axis=1)
        pdf.columns = pd.MultiIndex.from_product([sensors, axes],
                                                 names=['Sensor', 'Axis'])
        fig, ax = plt.subplots(figsize=(10, 8), nrows=3, ncols=1, sharex=True)
        self.plot_acc(pdf, None, [('acc', 'x'), ('acc', 'y'), ('acc', 'z')],
                 ax=ax[0], title='Phone accelerometer')
        return pdf

    def find_common_period(self):

        # Find commom period
        b = np.max((self.sensor_dataframe['OH1', 'ACC'].index[0], self.sensor_dataframe['H10', 'ACC'].index[0],
                    self.pdf.index[0]))
        e = np.min((self.sensor_dataframe['OH1', 'ACC'].index[-1], self.sensor_dataframe['H10', 'ACC'].index[-1],
                    self.pdf.index[-1]))

        fig, ax = plt.subplots(figsize=(10, 6), nrows=2, ncols=1)

        x, cdf = {}, {}
        cdf['phone'] = self.pdf[(self.pdf.index > b) & (self.pdf.index < e)]
        x['phone'] = np.sqrt(
            cdf['phone'][('acc', 'x')] ** 2 + cdf['phone'][('acc', 'y')] ** 2 +
            cdf['phone'][('acc', 'z')] ** 2)

        for key in ['OH1', 'H10']:
            cdf['phone'] = cdf['phone'].drop_duplicates()
            cdf['phone'] = cdf['phone'].reset_index(drop=True)
            cdf['phone'] = cdf['phone'].reset_index()
            self.sensor_dataframe[key, 'ACC'] = self.sensor_dataframe[key, 'ACC'].drop_duplicates()
            self.sensor_dataframe[key, 'ACC'] = self.sensor_dataframe[key, 'ACC'].reset_index(drop=True)
            self.sensor_dataframe[key, 'ACC'] = self.sensor_dataframe[key, 'ACC'].reset_index()
            cdf[key] = self.sensor_dataframe[key, 'ACC'].reindex(index=cdf['phone'].index,
                                              method='nearest')
            x[key] = np.sqrt(
                cdf[key]['[ns];X'] ** 2 + cdf[key]['[mg];Y'] ** 2 + cdf[key]['[mg];Z'] ** 2)
        logging.info(
            f'Phone cut {cdf["phone"].shape[0]} OH1 cut {cdf["OH1"].shape[0]} H10 cut {cdf["H10"].shape[0]}')

        bh, ah = sig.butter(4, 1 / (100 / 2), 'highpass')
        bl, al = sig.butter(4, 10 / (100 / 2), 'lowpass')

        for i in ['phone', 'OH1', 'H10']:
            # x[i] -= np.mean(x[i])
            x[i] = sig.filtfilt(bh, ah, x[i])
            x[i] = np.absolute(x[i])
            x[i] = sig.filtfilt(bl, al, x[i])
            ax[0].plot(x[i], label=i)
        ax[0].legend()

        offset = {}
        for pair in (('phone', 'OH1'), ('phone', 'H10'), ('H10', 'OH1')):
            l = ax[1].xcorr(x[pair[0]], x[pair[1]], maxlags=1000,
                            usevlines=False, label=f'{pair[0]} vs. {pair[1]}',
                            marker='.', linestyle='-')
            offset[pair] = l[0][l[1].argmax()]
            logging.info(f'Offset {pair[0]} vs. {pair[1]} is: {offset[pair]}')
        ax[1].legend()
        self.plot_sensors(self.pdf)

        for key in self.sensor_dataframe.keys():
            off = offset['phone', key[0]]
            logging.info(f'Sensor {key[0]} Data {key[1]} Offset: {off}')
            #print(self.sensor_dataframe[key].index)
            time_delta = timedelta(milliseconds=0.0 + off * 10)
            print(self.sensor_dataframe[key]["timestamp;sensor"])
            self.sensor_dataframe[key]['timestamp;sensor'] = pd.to_datetime(self.sensor_dataframe[key]['timestamp;sensor'],
                                             unit='ns').dt.round('1ms')

            self.sensor_dataframe[key]['adj_timestamp'] = self.sensor_dataframe[key]['timestamp;sensor'] + time_delta
            self.sensor_dataframe[key] = self.sensor_dataframe[key].set_index('adj_timestamp')

        for key in self.sensor_dataframe.keys():
            idx = self.pdf.index[(self.pdf.index >= self.sensor_dataframe[key].index[0]) & (
            (self.pdf.index <= self.sensor_dataframe[key].index[-1]))]
            aidx = idx.union(self.sensor_dataframe[key].index)
            interp_df = self.sensor_dataframe[key].reindex(aidx).interpolate(method='linear')
            index_df = interp_df.loc[idx]
            for column in self.sensor_dataframe[key].columns:
                self.pdf[f'{key[0]}-{key[1]}', column] = index_df[column]

    def clean_sensor(self, sensor_df, idx):
        logging.info(f'Original number of samples: {sensor_df.shape[0]}')
        # Leave unique values per sample
        sensor_df = sensor_df.groupby('DT').mean()
        logging.info(
            f'After duplicate removal, number of samples: {sensor_df.shape[0]}')
        # Add missing data
        cidx = idx.union(sensor_df.index)
        sensor_df = sensor_df.reindex(cidx).interpolate(method='linear')
        sensor_df = sensor_df.loc[idx]
        logging.info(
            f'After interpolation to 100 samples/sec, number of samples: {sensor_df.shape[0]}')
        return sensor_df

    def plot_sensors(self, df):
        fig, ax = plt.subplots(figsize=(10, 8), nrows=4, ncols=1,
                               sharex=True)
        self.plot_acc(df, ('OH1', 'ACC'), ['x', 'y', 'z'], ax[0],
                 'OH1 accelerometer')
        self.plot_acc(df, ('H10', 'ACC'), ['x', 'y', 'z'], ax[1],
                 'H10 accelerometer')
        self.plot_acc(df, ('OH1', 'PPG'), ['ch1', 'ch2', 'ch3'], ax[2],
                 'OH1 PPG')
        self.plot_acc(df, ('H10', 'ECG'), ['ecg'], ax[3], 'H10 ECG')

    def plot_acc(self, df, key, fields, ax, title=''):
        if key is None:
            cdf = df
        else:
            if key in df:
                cdf = df[key]
            else:
                logging.info(f'Key ({key}) does not exist')
                return
        for field in fields:
            ax.plot(cdf.index, cdf[field], label=field)
        ax.legend()
        ax.set_title(title)
        plt.show()

