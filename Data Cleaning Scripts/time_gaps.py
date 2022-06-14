import time

time_format = "1/7/2013 0:00:00"

time_position = time_format.find(" ")

time_format = time_format.replace("/", " ")

clock = time_format[time_position + 1:len(time_format)]

seconds = clock.count(":")

fred = clock.replace(":", " ")

print(time_format)
print(clock)
