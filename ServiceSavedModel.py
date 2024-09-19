
import csv

def appendCsvRow(path, fieldnames, rowDict):
    with open(path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(rowDict)

"""
def appendCsvRow(path, row):
    with open(path,'a') as fd:
        fd.write(row)
"""