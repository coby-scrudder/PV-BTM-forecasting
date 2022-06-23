import os
import csv
from function_calls import weather_station_from_id
import time

time_start = time.time()
time_last = time_start

weather_station_csv = 'site_weather_coordinates.csv'

PV_data_directory = 'D:/REU Research/PV Data/'
weather_directory = 'D:/REU Research/Weather Data Interpolated/'
write_combined_directory = 'D:/REU Research/Combined Data/'

count = 0

for x in os.listdir(PV_data_directory):
    for file in os.listdir(PV_data_directory + x + '/'):
        count = 0
        if file[-3:] == 'csv' and file[-7:-4] != 'row':
            print(file)
            year, system_id = file.split('_')
            _, year = year.split('=')
            if int(year) > 2020:
                break
            _, system_id = system_id.split('=')
            system_id, _ = system_id.split('.')
            lat, long = weather_station_from_id(weather_station_csv, system_id)
            file_name = f'{float(lat):.2f}_{float(long):.2f}_{year}.csv'
            with open(weather_directory + file_name, 'r') as csv_weather:
                csv_weather_reader = csv.reader(csv_weather)
                weather_list = []
                for line in csv_weather_reader:
                    weather_list.append(line)
            write_file = f'{write_combined_directory}id_{system_id}_{year}.csv'
            with open(write_file, 'w', newline='') as csv_write:
                csv_writer = csv.writer(csv_write)
                with open(PV_data_directory + x + '/' + file, 'r') as csv_read:
                    csv_reader = csv.reader(csv_read)
                    for row in csv_reader:
                        headers = [row[0], 'Power Measured', 'Latitude', 'Longitude']
                        for header in row[1:6]:
                            headers.append(header)
                        break
                    for row in csv_reader:
                        measured_on, test_year, test_month, test_day, test_hour, test_minute, power = row
                        for index, line in enumerate(weather_list):
                            if index == 0 and count == 0:
                                for element in line[5:]:
                                    headers.append(element)
                                csv_writer.writerow(headers)
                                count += 1
                            year, month, day, hour, minute = line[:5]
                            if year == test_year and minute == test_minute and month == test_month and hour == test_hour and day == test_day:
                                weather_list.remove(line)
                                new_row = [measured_on, power, lat, long,test_year, test_month, test_day, test_hour, test_minute]
                                for element in line[5:]:
                                    new_row.append(element)
                                csv_writer.writerow(new_row)
                                if count % 100 == 0:
                                    print(new_row)
                                    time_now = time.time()
                                    print(f'Time Elapsed = {(time_now - time_start):.2f} seconds')
                                    print(f'Time for iteration = {(time_now - time_last):.2f} seconds')
                                    time_last = time_now
                                count += 1
