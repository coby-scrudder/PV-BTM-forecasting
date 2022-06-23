import os
import csv
import numpy as np

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
                    hour, minute, seconds = time_elements
                    hour = str(int(hour))
                    minute = int(minute)
                    seconds = int(seconds)
                    # print('Hour:', hour, 'Minute:', minute, time_elements)
                    output_write = [row[0], year, month, day, hour, str(minute), row[1]]
                    if minute == 0 or minute == 15 or minute == 30 or minute == 45:
                        csv_writer.writerow(output_write)


def combine_years_and_cut(directory, output_csv, keep_column):
    with open(output_csv, "w+", newline='') as csv_file_old:
        csvwriter = csv.writer(csv_file_old)
        csvwriter.writerow(['measured_on', keep_column])
        for file in os.listdir(directory):
            with open(directory + '/' + file, "r") as day_file:
                csvreader_1 = csv.reader(day_file)
                for row in csvreader_1:
                    columns = row
                    break
                index_keep = columns.index(keep_column)
                new_rows = []
                for row in csvreader_1:
                    new_rows.append([row[0], row[index_keep]])
                    csvwriter.writerows(new_rows)
                    new_rows = []

    # with open(input_csv, "r", newline='') as csv_file_old:
    #     with open(output_csv, 'a', newline='') as csv_file_new:
    #         csvreader_2 = csv.reader(csv_file_old)
    #         csvwriter = csv.writer(csv_file_new)
    #         for row in csvreader_2:
    #             columns = row
    #             break
    #         index_keep = columns.index(keep_column)
    #         new_rows = []
    #         csvwriter.writerow(['measured_on', keep_column])
    #         for row in csvreader_2:
    #             row_cut = []
    #             if row != []:
    #                 try:
    #                     row_cut.append([row[0], row[index_keep]])
    #                 except IndexError:
    #                     print(row)
    #                     print(index_keep)
    #                     print(len(row))
    #                     sys.exit()
    #                 new_rows.append(row_cut)
    #         for row in new_rows:
    #             csvwriter.writerows(row)
                # if new_rows.index(row) < 10000:
                #     if new_rows.index(row) % 10 == 0:
                #         plt.plot(new_rows.index(row), float(row[0][1]), 'bo')
                # else:
                #     break
    #         plt.xlabel('Date')
    #         plt.ylabel('Power')
    # plt.show()

def find_column_to_keep(csv_file, system_id, year):
    row_name = f'system_id={system_id}/year={year}'
    with open(csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[0] == row_name:
                return row[2]


def average_lists(list1, list2):
    '''Given two input lists with float values as string datatype,
    this returns the average of the values of the lists'''
    array1 = np.array(list1).astype(float)
    array2 = np.array(list2).astype(float)

    array3 = (array1 + array2) / 2

    list3 = list(array3.astype(str))
    return list3


def weather_station_from_id(location_csv, systemid):
    '''Returns latitude, longitude'''
    with open(location_csv, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        rows = []
        for row in csv_reader:
            rows.append(row)
        for x in rows:
            if x[0] == str(systemid):
                return x[1], x[2]
