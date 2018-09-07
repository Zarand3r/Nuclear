import numpy as np
import os
import ntpath
import glob
import shutil
from PIL import Image

#Creates a data directory in CNN3
#This directory contains nuclei converted to jpg files
#The nuclei images are organized into 4 subdirectories, corresponding to the 4 morphology classes. 
LABELS = ["Acute_Scar", "Micronuclei", "Subacute_Scar", "Unscarred"]
SOURCE_DIRECTORY = "../../output/nuclei" 
OUTPUT_DIRECTORY = "../ImageData"

def initialize(dataDir):
	#Create a new data directory, with a subdirectory for each label
	os.makedirs(os.path.join(dataDir), exist_ok=True) 
	for label in LABELS:
		os.makedirs(os.path.join(dataDir, label), exist_ok=True) 

def convert_to_jpg(scan_name, image_path, label, output_path):
	#opens tiff image, converts to jpg, saves to output path
	fname = ntpath.basename(image_path)
	name = os.path.splitext(fname)[0]
	extension = os.path.splitext(fname)[1].lower()
	if extension == ".tiff":
		outfile = os.path.join(output_path, label, scan_name + name + ".jpg")
		if os.path.isfile(outfile):
			print ("A jpeg file already exists for %s" % name)
			# If a jpeg is *NOT* present, create one from the tiff.
		else:
			im = Image.open(image_path)
			print ("Generating jpeg for %s" % name)
			im.thumbnail(im.size)
			im.save(outfile, "JPEG", quality=100)

#Make a function that creates the labels.txt file in data_scripts that build_image_data depends on
def make_label_file():
	label_file = open("label_names.txt", "w")
	for label in LABELS:
		label_file.write(label+'\n')
	label_file.close()


def write_data(source, output_path):
	initialize(output_path)
	dirList = []
	scan_name = ""
	for dirName, subdirList, fileList in os.walk(source):
		print('Found directory: %s' % dirName)
		label = ntpath.basename(dirName)
		if dirName == source:
			dirList = subdirList
		else:
			if ntpath.basename(dirName) in dirList:
				scan_name = ntpath.basename(dirName)
		if label in LABELS:
			for fname in fileList:
				image_path = os.path.join(dirName, fname)
				convert_to_jpg(scan_name, image_path, label, output_path)


if __name__ == "__main__":
	make_label_file()
	# write_data(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)
