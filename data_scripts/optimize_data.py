import numpy as np
import os
import ntpath
import glob
import shutil
import csv
import random
from PIL import Image
#modified version of ImageData repository
#reduce number of unscarred nuclei to avoid overclustering
#for scar nuclei, add different orientations as new data to tell the neural net that there is no association with where the scar is

DATA_INPUT = "../ImageData"
DUMP_DIRECTORY = "../Dump"
LABELS = ["Acute_Scar", "Micronuclei", "Subacute_Scar", "Unscarred"]

def parse(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def reduce(rootDir, ratio):
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if ntpath.basename(dirName) in LABELS:
			label = ntpath.basename(dirName)
			#reduce unscarred
			if label == "Unscarred":
				os.makedirs(os.path.join(DUMP_DIRECTORY, "Unscarred"), exist_ok = True) 
				for fname in fileList:
					if random.random() < ratio:
						shutil.move(os.path.join(dirName,fname), os.path.join(DUMP_DIRECTORY, label, fname))

def restore(rootDir):
	for dirName, subdirList, fileList in os.walk(DUMP_DIRECTORY):
		print('Found directory: %s' % dirName)
		if ntpath.basename(dirName) in LABELS:
			label = ntpath.basename(dirName)
			#reduce unscarred
			if label == "Unscarred":
				for fname in fileList:
					shutil.move(os.path.join(dirName,fname), os.path.join(rootDir, label, fname))

def rotate(rootDir, label):
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if ntpath.basename(dirName) == label:
			for fname in fileList:
				if fname == ".DS_Store":
					continue
				image_path = os.path.join(dirName, fname)
				picture = Image.open(image_path)
				image_id = parse(image_path, "Nucleus_", "_")
				new_path1 = image_path.replace(image_id, image_id+"(1)")
				new_path2 = image_path.replace(image_id, image_id+"(2)")
				new_path3 = image_path.replace(image_id, image_id+"(3)")
				picture.rotate(90).save(new_path1)
				picture.rotate(180).save(new_path2)
				picture.rotate(270).save(new_path3)


def optimize(rootDir, ratio = 0.5):
	# reduce(rootDir, ratio)
	rotate(rootDir, "Micronuclei")


if __name__ == "__main__":
	optimize(DATA_INPUT)


