import os
import csv
import pandas
import matplotlib.pyplot as plt



def combine_years_and_cut(directory, temporary_csv, output_csv, keep_column):
    count = 0
    with open(temporary_csv, "w+", newline='') as csv_file_old:
        csvwriter = csv.writer(csv_file_old)
        for file in os.listdir(directory):
            with open(directory + '/' + file, "r") as day_file:
                csvreader_1 = csv.reader(day_file)
                if count != 0:
                    next(csvreader_1)
                count += 1
                for row in csvreader_1:
                    csvwriter.writerow(row)

    with open(temporary_csv, "r", newline='') as csv_file_old:
        with open(output_csv, 'a', newline='') as csv_file_new:
            csvreader_2 = csv.reader(csv_file_old)
            csvwriter = csv.writer(csv_file_new)
            for row in csvreader_2:
                columns = row
                break
            index_keep = columns.index(keep_column)
            new_rows = []
            csvwriter.writerow(['measured_on', keep_column])
            for row in csvreader_2:
                row_cut = []
                row_cut.append([row[0], row[index_keep]])
                new_rows.append(row_cut)
            for row in new_rows:
                csvwriter.writerows(row)
                # if new_rows.index(row) < 10000:
                #     if new_rows.index(row) % 10 == 0:
                #         plt.plot(new_rows.index(row), float(row[0][1]), 'bo')
                # else:
                #     break
    #         plt.xlabel('Date')
    #         plt.ylabel('Power')
    # plt.show()
