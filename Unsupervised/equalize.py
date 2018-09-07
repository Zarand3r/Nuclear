import numpy as np
import os
import sys
import random
import math
import ntpath
import csv
import glob
import shutil

#This program reorgaanizes a specific nuclei directory, creating sorted subfolders labeled by nuclear morphology class


def parse(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def get_header(original_csv):
	input_file = open(original_csv, "r")
	reader = csv.reader(input_file)
	return next(reader)


def get_nucleus(scanID, nucID, original_csv):
	input_file = open(original_csv, "r")
	reader = csv.reader(input_file)
	for row in reader:
		if (str(scanID) == row[0] and str(nucID) == row[1]):
			return row

def reduce(rootDir, depth, original_csv, new_csv):
	scanID = 0
	dirList = []
	labels = ["Acute Scar", "Micronuclei", "Subacute Scar", "Unscarred"]
	output_file = open(new_csv, 'w')
	writer = csv.writer(output_file)

	header = get_header(original_csv)
	writer.writerow(header)
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		elif ntpath.basename(dirName) in dirList:
			scanID = dirList.index(ntpath.basename(dirName))+1
			if scanID > depth:
				break;
		# to exclude the ??? unlabeled nuclei, use this lin --> elif ntpath.basename(dirName) in labels:
		else: # this includes the ??? unlabeled nuclei 
			for fname in fileList:
				if fname != ".DS_Store":
					nucID = int(parse(fname, "_", "_"))
					label = ntpath.basename(dirName)
					if label == "Unscarred":
						if random.random() < 0.9:
							continue;
					row = get_nucleus(scanID, nucID, original_csv)
					writer.writerow(row)

if __name__ == "__main__":
	reduce('../output/nuclei', 20, '../output/Measurements/Nucleus_BAF_Nuclei.csv', 'reduced.csv')


