import os

directory = "D:/REU Research/PV Data/"

for x in os.listdir(directory):  # system_id
    for a in os.listdir(directory + x):  # lists year
        for b in os.listdir(directory + x + "/" + a):  # lists month
            if b[-3]+b[-2]+b[-1] != "csv":
                print(directory + x + "/" + a + "/" + b)

