import os

directory = "D:/REU Research/PV Data/"

systems_missed = [1203, 1208, 1262, 1271, 1274, 1283, 1306, 1308, 1309,
                  1310, 1332, 1347, 1352, 1355,
                  1368, 1416, 1422, 1423, 3, 33, 4, 50, 51]

system_ids = []

for x in systems_missed:
    system_ids.append(directory + "system_id=" + str(x) + "/")

for system_id in system_ids:
    for csv_file in os.listdir(system_id):
        year = csv_file[-14:-10]
        if not os.path.exists(system_id + f"year={year}" + "/"):
            os.mkdir(system_id + f"year={year}" + "/")
        os.rename(system_id + csv_file, system_id + f"year={year}" + "/" + csv_file)

