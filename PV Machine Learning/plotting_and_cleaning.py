import os
import csv
import statistics
import matplotlib.pyplot as plt

data_directory = 'D:/REU Research/Trial Combined Data/'

count = 0
prev_day = 9

for file in os.listdir(data_directory):
    dates = []
    powers = []
    print(file)
    with open(data_directory + file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        day_powers = [0]
        for i, row in enumerate(csv_reader):
            day = int(row[6])
            try:
                power = float(row[1])
            except:
                power = 0
            if i == 0:
                prev_day = day
            if day == prev_day:
                day_powers.append(power)
            else:
                try:
                    powers.append(max(day_powers))
                except ValueError:
                    print(row[0])
                    sys.exit()
                dates.append(row[0])
                day_powers = [0]
            prev_day = day
    # plot_dates = []
    # plot_powers = []
    # for i, x in enumerate(powers):
    #     if x > statistics.mean(powers) / 1.5:
    #         plot_powers.append(x)
    #         plot_dates.append(dates[i])
    plt.subplot(2, 3, (count % 6) + 1)
    plt.plot_date(dates, powers, fmt='b-')
    plt.tick_params(bottom=False)
    plt.title(file)
    if count % 6 == 5:
        plt.show()
    count += 1
