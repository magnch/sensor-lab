# Split csv file into five parts, withh 600 lines each.
import csv
import numpy as np

# Define file paths
filename = "./csv/lab3/reflektans_rapport.csv"
output_file = filename[:-4] + "_måling_"
length = 300   # Number of lines in each part
offset = 100

# Read data from file
data = []
with open(filename, "r") as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

print(len(data))

# Split data into six parts
for i in range(6):
    if i == 5:
        part = data[i * length:]
    else:
        part = data[i * length:(i + 1) * length]
    part_filename = output_file + str(i + 1) + ".csv"
    with open(part_filename, "w", newline="") as file:
        writer = csv.writer(file)
        #Write without lines between rows
        writer.writerows(part)
    print(f"Data saved to {part_filename}")