import csv
import os
from function_calls import find_column_to_keep

directory = 'D:/REU Research/PV Data/'

count = 0

for x in os.listdir(directory):
    _, system_id = x.split('=')
    for y in os.listdir(directory + x + '/'):
        _, year = y.split('=')
        keep_column = find_column_to_keep('power_parameters.csv', system_id, year)
        for z in os.listdir(directory + x + '/' + y + '/'):
            remove = 0
            csv_directory = directory + x + '/' + y + '/' + z
            if z[-4:] == ".csv":
                count += 1
                with open(csv_directory, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        if keep_column not in row:
                            count += 1
                            remove = 1
                        break
                if remove == 1:
                    os.remove(csv_directory)
                    print(f"{count}/3884: Deleted ", csv_directory)
