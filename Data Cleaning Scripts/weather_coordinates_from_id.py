import csv

def id_to_coord(system_id, weather_location_csv):
    with open(weather_location_csv, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        for row in csv_reader:
            if int(row[0]) == system_id:
                _, lat, long = row
                lat = f'{float(lat):.2f}'
                long = f'{float(long):.2f}'
                return lat, long


weather_csv = 'site_weather_coordinates.csv'

print(id_to_coord(4, weather_csv))
