import pandas as pd
import numpy as np
import csv
from itertools import chain

final_user_study_path = "/Users/saqibali/PycharmProjects/DTW/FinalUserStudy"


def generate_gesture_intervals(i):
    """
    Creates tuples of start and end time stamps for each gesture in a participant i
    :return: List[List(int, int)] where each inner list is a list of intervals
    """

    #  Convert log file to data frame
    file_name = final_user_study_path + "/P" + str(i) + "/DataCollection/logs/log.csv"
    log_df = pd.read_csv(file_name)

    #  list of intervals for each gesture in a specific participant i
    swipe_right_intervals = []
    swipe_left_intervals = []
    whats_up_intervals = []
    nod_intervals = []

    #  filling in the interval lists for each gesture in a participant i
    for index, row in log_df.iterrows():
        if row['gesture'] == 'Nod' and row['label'] == 'y': nod_intervals.append((int(row['start']), int(row['end'])))
        elif row['gesture'] == 'Swipe Left' and row['label'] == 'y': swipe_left_intervals.append((int(row['start']), int(row['end'])))
        elif row['gesture'] == 'Swipe Right' and row['label'] == 'y': swipe_right_intervals.append((int(row['start']), int(row['end'])))
        elif row['gesture'] == 'Whats Up' and row['label'] == 'y': whats_up_intervals.append((int(row['start']), int(row['end'])))
    return nod_intervals, swipe_left_intervals, swipe_left_intervals, whats_up_intervals


def fill_gyro_and_acc_data(interval, merge):
    """

    :param interval: interval for a specific gesture
    :param merge: merge data frame with 2 hours of gesture sensor data
    :return: List[List[int]] of each class of gestures' sensor data for each participant
    """

    #  Outer gesture sensor list for all instances of that gesture
    gesture_gyro = []
    gesture_acc = []
    for i in interval:
        gyro = []
        acc = []
        start = np.abs(merge.iloc[:, 1] - i[0]).argmin()
        end = np.abs(merge.iloc[:, 1] - i[1]).argmin()
        for j in range(start, end):
            gyro.append((merge.iloc[j, 2], merge.iloc[j, 3], merge.iloc[j, 4])) \
                if merge.iloc[j, 0] == 'Gyroscope' else acc.append((merge.iloc[j, 2], merge.iloc[j, 3],
                                                                    merge.iloc[j, 4]))
        gesture_gyro.append(gyro)
        gesture_acc.append(acc)
    return [gesture_gyro, gesture_acc]


def generate_gesture_lists(i):
    nod_intervals, swipe_left_intervals, swipe_right_intervals, whats_up_intervals = generate_gesture_intervals(i)

    merge = pd.read_csv(final_user_study_path + "/P" + str(i) + "/DataCollection/data/merge.csv")

    nod_gyro, nod_acc = fill_gyro_and_acc_data(nod_intervals, merge)
    swipe_right_gyro, swipe_right_acc = fill_gyro_and_acc_data(swipe_right_intervals, merge)
    swipe_left_gyro, swipe_left_acc = fill_gyro_and_acc_data(swipe_left_intervals, merge)
    whats_up_gyro, whats_up_acc = fill_gyro_and_acc_data(whats_up_intervals, merge)

    return nod_gyro, nod_acc, swipe_right_gyro, swipe_right_acc, swipe_left_gyro, swipe_right_acc, whats_up_gyro, whats_up_acc


def convert_gesture_lists_to_files(i):
    sensor_data = generate_gesture_lists(i)
    names = ['nod_gyro', 'nod_acc', 'swipe_right_gyro', 'swipe_right_acc', 'swipe_left_gyro', 'swipe_right_acc',
             'whats_up_gyro', 'whats_up_acc']
    assert(len(sensor_data) == len(names))
    for j in range(len(sensor_data)):
        with open(names[j] + "P" + str(i) + ".csv", "wb") as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y', 'z'])
            writer.writerows(list(chain.from_iterable(sensor_data[j])))

for i in range(2, 12):
    convert_gesture_lists_to_files(i)