import os

directory = 'D:/REU Research/PV Data Every 15/'

for x in os.listdir(directory):
    for file in os.listdir(directory + x + '/'):
        if file[-3:] == 'csv':
            os.remove(directory + x + '/' + file)