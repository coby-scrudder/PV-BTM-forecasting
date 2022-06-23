import csv


def find_column_to_keep(csv_file, system_id, year):
    row_name = f'system_id={system_id}/year={year}'
    with open(csv_file, 'r', encoding='utf-8-sig') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[0] == row_name:
                return row[2]
