import os
import sys
import csv
import pandas
import matplotlib.pyplot as plt
from function_calls import combine_years_and_cut, find_column_to_keep, split_date_on_csv

directory = 'D:/REU Research/PV Data/'

for x in os.listdir(directory):
    _, system_id = x.split('=')
    print(x)
    for y in os.listdir(directory + x + '/'):
        _, year = y.split('=')
        print('year =', year)
        output_csv_every_data_point = directory + x + '/' + f'year={year}_id={system_id}_every_row.csv'
        output_csv = directory + x + '/' + f'year={year}_id={system_id}.csv'
        keep_column = find_column_to_keep('power_parameters.csv', system_id, year)
        directory_year = directory + x + '/' + y + '/'
        combine_years_and_cut(directory_year, output_csv_every_data_point, keep_column)
        split_date_on_csv(output_csv_every_data_point, output_csv)
