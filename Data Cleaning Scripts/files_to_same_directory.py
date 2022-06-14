import os

directory = "D:/REU Research/PV Data/"

for x in os.listdir(directory):  # system_id
    for a in os.listdir(directory + x):  # lists year
        for b in os.listdir(directory + x + "/" + a):  # lists month
            if b[-3]+b[-2]+b[-1] == 'csv':
                break
            if b == "__temp__":
                break
            for c in os.listdir(directory + x + "/" + a + "/" + b):  # lists day
                for d in os.listdir(directory + x + "/" + a + "/" + b + "/" + c):  # lists csv in a day
                    print(d)
                    if d == "__temp__":
                        print("True")
                        for e in os.listdir(directory + x + "/" + a + "/" + b + "/" + c + "/" + d):
                            os.remove(directory + x + "/" + a + "/" + b + "/" + c + "/" + d + "/" + e)
                        os.rmdir(directory + x + "/" + a + "/" + b + "/" + c + "/" + d)
                        break
                    new_directory = directory + x + "/" + a + "/"
                    old_directory = directory + x + "/" + a + "/" + b + "/" + c + "/" + d
                    new_directory = new_directory + d
                    os.rename(old_directory, new_directory)
                    try:
                        os.rmdir(directory + x + "/" + a + "/" + b + "/" + c)
                    except:
                        break
            try:
                os.rmdir(directory + x + "/" + a + "/" + b)
            except:
                break
