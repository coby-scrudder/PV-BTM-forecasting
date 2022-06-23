import os
import csv
import pandas
import matplotlib.pyplot as plt


def split_date_on_csv(input_csv, output_csv):
    with open(input_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        with open(output_csv, 'w', newline='') as csv_output:
            csv_writer = csv.writer(csv_output)
            for index, row in enumerate(csv_reader):
                if index == 0:
                    output_write = [row[0], 'Year', 'Month', 'Day', 'Hour', 'Minute', row[1]]
                    csv_writer.writerow(output_write)
                else:
                    date = row[0]
                    calendar = date[:10]
                    date_elements = calendar.split('-')
                    year, month, day = date_elements
                    month = str(int(month))
                    day = str(int(day))
                    # print('Year:', year, 'Month:', month, 'Day:', day, calendar)
                    time = date[-8:]
                    time_elements = time.split(':')
                    hour, minute, _ = time_elements
                    hour = str(int(hour))
                    minute = int(minute)
                    # print('Hour:', hour, 'Minute:', minute, time_elements)
                    output_write = [row[0], year, month, day, hour, str(minute), row[1]]
                    if minute == 0 or minute == 15 or minute == 30:
                        csv_writer.writerow(output_write)
