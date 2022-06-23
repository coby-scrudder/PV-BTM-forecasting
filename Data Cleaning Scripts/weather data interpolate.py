import csv
import os
from function_calls import average_lists

starting_directory = 'D:/REU Research/Weather Data Kelvin/'
ending_directory = 'D:/REU Research/Weather Data Interpolated/'

for file in os.listdir(starting_directory):
    with open(starting_directory + file, 'r') as csv_file:
        with open(ending_directory + file, 'w', newline='') as csv_write:
            csv_writer = csv.writer(csv_write)
            csv_reader = csv.reader(csv_file)
            rows = []
            new_rows = []
            for row in csv_reader:
                csv_writer.writerow(row)
                break
            for row in csv_reader:
                rows.append(row)
            for x in range(len(rows) - 1):
                interpolated_list_without_time = average_lists(rows[x][5:], rows[x + 1][5:])
                if float(rows[x][4]) < float(rows[x+1][4]): # return 15 minutes
                    minute = 15
                else:
                    minute = 45
                new_rows.append(rows[x])
                interpolated_list = rows[x][:4]
                interpolated_list.append(str(minute))
                for x in interpolated_list_without_time:
                    interpolated_list.append(x)
                new_rows.append(interpolated_list)
            new_rows.append(rows[-1])
            for row in new_rows:
                csv_writer.writerow(row)
