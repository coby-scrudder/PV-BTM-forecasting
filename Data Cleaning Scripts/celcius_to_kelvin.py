import csv
import os

starting_directory = 'D:/REU Research/Weather Data Raw/'
ending_directory = 'D:/REU Research/Weather Data Kelvin/'

for x in os.listdir(starting_directory):
    for y in os.listdir(starting_directory + x + '/'):
        for file in os.listdir(starting_directory + x + '/' + y + '/'):
            csv_dir = starting_directory + x + '/' + y + '/' + file
            _, lat, long, year = file.split('_')
            year = year[:4]
            new_csv = ending_directory + f'{lat}_{long}_{year}.csv'
            with open(csv_dir, 'r') as csv_read:
                csv_reader = csv.reader(csv_read)
                with open(new_csv, 'w', newline='') as csv_write:
                    csv_writer = csv.writer(csv_write)
                    next(csv_reader)
                    next(csv_reader)
                    for row in csv_reader:
                        dew_point_index = row.index('Dew Point')
                        temp_index = row.index('Temperature')
                        fill_flag_index = row.index('Fill Flag')
                        row.remove('Fill Flag')
                        csv_writer.writerow(row)
                        break
                    for row in csv_reader:
                        temp_new = float(row[temp_index]) + 273.15
                        dew_new = float(row[dew_point_index]) + 273.15
                        row_new = [row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9],
                                   row[10], row[11], dew_new, row[13], row[15], row[16], row[17], row[18], row[19],
                                   temp_new, row[21], row[22], row[23]]
                        csv_writer.writerow(row_new)
