import csv
import json
import os

directory = "D:/REU Research/PV Data/system_metadata/"
directory_csv = "D:/REU Research/PV Data/"
csv_file = directory_csv + "systems_data.csv"

count = 0

for filename in os.listdir(directory):
    filepath = directory + filename
    with open(filepath, 'r') as f:
        json_dict = json.load(f)
        system = json_dict["System"]
        system.update(json_dict["Site"])
        system.update(json_dict["Mount"]["Mount 0"])
        print(system)

        list_keys = []
        for key in system:
            list_keys.append(key)

        with open(csv_file, "a", newline='') as csvfile:
            fieldnames = list_keys
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if count == 0:
                writer.writeheader()
                count += 1
            writer.writerow(system)

