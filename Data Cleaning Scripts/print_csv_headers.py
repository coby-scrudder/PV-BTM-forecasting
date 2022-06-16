import csv
import os

directory = "D:/REU Research/PV Data/"

list_of_headers = []

for x in os.listdir(directory):
    for y in os.listdir(directory + x + '/'):
        count = 0
        for z in os.listdir(directory + x + '/' + y + '/'):
            if count == 0:
                with open(directory + x + '/' + y + '/' + z) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    count_row = 0
                    list_of_column_names = []
                    for row in csv_reader:
                        row.insert(0, x + '/' + y)
                        if count == 0:
                            list_of_column_names.append(row)
                        count += 1
                    list_of_headers.append(list_of_column_names)

directory_csv = "headers.csv"
with open(directory_csv, "w", newline='') as csv_file:
    for i in list_of_headers:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerows(i)

