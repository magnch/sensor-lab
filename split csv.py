# Split csv file into five parts, withh 600 lines each.

import sys
import os
import csv

# Define file paths
filename = "./csv/transmittans_120s.csv"
output_file = "./csv/transmittans_måling_"

# Read data from file
data = []
with open(filename, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

data = data[592:]
print(data)

# Split data into five parts
for i in range(5):
    part = data[i * 600:(i + 1) * 600]
    part_filename = output_file + str(i + 1) + ".csv"
    with open(part_filename, "w", newline="") as file:
        writer = csv.writer(file)
        #Write without lines between rows
        writer.writerows(part)
    print(f"Data saved to {part_filename}")