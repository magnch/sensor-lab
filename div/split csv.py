# Split csv file into five parts, withh 600 lines each.

import sys
import os
import csv

# Define file paths
filename = "./csv/lab3/transmittans_rapport.csv"
output_file = filename[:-4] + "_måling_"
length = 300   # Number of lines in each part
offset = 293

# Read data from file
data = []
with open(filename, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

data = data[offset:]
print(len(data))

# Split data into five parts
for i in range(5):
    part = data[i * length:(i + 1) * length]
    part_filename = output_file + str(i + 1) + ".csv"
    with open(part_filename, "w", newline="") as file:
        writer = csv.writer(file)
        #Write without lines between rows
        writer.writerows(part)
    print(f"Data saved to {part_filename}")