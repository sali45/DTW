import random, math
import os
import csv


def k_fold(myfile, myseed=11109, k=20):
    # Load data
    data = open(myfile).readlines()

    # Shuffle input
    random.seed=myseed
    random.shuffle(data)

    # Compute partition size given input k
    len_part=int(math.ceil(len(data)/float(k)))

    # Create one partition per fold
    train={}
    test={}
    for ii in range(k):
        test[ii]  = data[ii*len_part:ii*len_part+len_part]
        train[ii] = [jj for jj in data if jj not in test[ii]]

    return train, test


data_dir = "/Users/saqibali/PycharmProjects/MagicTheGathering/Data/WhatsUp/Gyro"
x_train = []
x_test = []
y_train = []
y_test = []

fout=open("x_train.csv","a")
for num in range(1,201):
    for line in open("sh"+str(num)+".csv"):
         fout.write(line)
fout.close()
# for filename in os.listdir(data_dir):
#     x_train.append(k_fold(os.path.join(data_dir, filename))[0])
#     x_test.append(k_fold(os.path.join(data_dir, filename))[1])
#
#
# with open("x_train_whats_up_gyro", 'wb') as my_file:
#     wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
#     wr.writerow(x_train)
#
# with open("x_test_whats_up_gyro", 'wb') as my_file:
#     wr = csv.writer(my_file, quoting=csv.QUOTE_ALL)
#     wr.writerow(x_test)
