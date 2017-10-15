import pandas
import numpy as np

final_user_study_path = "/Users/saqibali/PycharmProjects/DTW/FinalUserStudy"

def generate_gesture_intervals(i):
    """

    :return:
    """
    log_df = pandas.read_csv(final_user_study_path + "/P" + str(i) + "/DataCollection/logs/log.csv")

    nod_intervals = []
    swipe_right_intervals = []
    swipe_left_intervals = []
    whats_up_intervals = []

    for index, row in log_df.iterrows():
        if row['gesture'] == 'Nod' and row['label'] == 'y':
            nod_intervals.append((row['start'], row['end']))
        elif row['gesture'] == 'Swipe Left' and row['label'] == 'y':
            swipe_left_intervals.append((row['start'], row['end']))
        elif row['gesture'] == 'Swipe Right' and row['label'] == 'y':
            swipe_right_intervals.append((row['start'], row['end']))
        elif row['gesture'] == 'Whats Up' and row['label'] == 'y':
            whats_up_intervals.append((row['start'], row['end']))
    return nod_intervals, swipe_left_intervals, swipe_left_intervals, whats_up_intervals


def fill_gyro_and_acc_data(interval, merge):
    gyro = []
    acc = []
    for i in interval:
        start = np.abs(merge.iloc[:, 1] - i[0]).argmin()
        end = np.abs(merge.iloc[:, 1] - i[1]).argmin()
        for j in range(start, end):
            gyro.append((merge.iloc[j, 2], merge.iloc[j, 3], merge.iloc[j, 4])) if merge[merge.iloc[j, 1]] == 'Gyroscope' \
                else acc.append((merge.iloc[j, 2], merge.iloc[j, 3], merge.iloc[j, 4]))iloc
        return gyro, acc


def generate_gesture_files(i):
    nod_intervals, swipe_left_intervals, swipe_right_intervals, whats_up_intervals = generate_gesture_intervals(i)

    merge = pandas.read_csv(final_user_study_path + "/P" + str(i) + "/DataCollection/data/merge.csv")
    nod_gyro, nod_acc = fill_gyro_and_acc_data(nod_intervals, merge)
    swipe_right_gyro, swipe_right_acc = fill_gyro_and_acc_data(swipe_right_intervals, merge)
    swipe_left_gyro, swipe_left_acc = fill_gyro_and_acc_data(swipe_left_intervals, merge)
    whats_up_gyro, whats_up_acc = fill_gyro_and_acc_data(whats_up_intervals, merge)

    return nod_gyro, nod_acc, swipe_right_gyro, swipe_right_acc, swipe_left_gyro, swipe_right_acc, whats_up_gyro, whats_up_acc

generate_gesture_files(2)
