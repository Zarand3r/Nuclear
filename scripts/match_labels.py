import numpy as np
import os
import sys
import random
import psd_parse_tools as ppt
import math
import ntpath
import csv
import matplotlib.pyplot as plt
import glob
import shutil



#This program reorgaanizes a specific nuclei directory, creating sorted subfolders labeled by nuclear morphology class


def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


#Creates 4 subfolders (for each nuclear morphology label) in each nuclei folder. 
def create(dirName):
	os.makedirs(os.path.join(dirName, "Micronuclei"), exist_ok=True) 
	os.makedirs(os.path.join(dirName, "Acute_Scar"), exist_ok=True) 
	os.makedirs(os.path.join(dirName, "Subacute_Scar"), exist_ok=True) 
	os.makedirs(os.path.join(dirName, "Unscarred"), exist_ok=True)

#Calculates distance between two coordinates
def distance(x1,y1,x2,y2):
    sq1 = (x1-x2)*(x1-x2)
    sq2 = (y1-y2)*(y1-y2)
    return math.sqrt(sq1 + sq2)

def coordinates(scanID, nucID):
	csv_file = open('../output/Measurements/Nucleus_BAF_Nuclei.csv', "r")
	reader = csv.reader(csv_file, delimiter=",")
	for row in reader:
		if (scanID == row[0] and nucID == row[1]):
			return (float(row[3]), float(row[4]))

#Accepts scan id and nucleus id as parameters
#Iterates through csv to find matching row and get location data
#Iterates through label list to find closest location match
def match(scanID, nucID):
	(x,y) = coordinates(scanID, nucID)
	score = 100
	label = "???"
	image_path = glob.glob(os.path.join("../psd/", "*.psd"))[int(scanID)-1]
	for key,value in ppt.getPositions(image_path).items():
	    for pos in value:
	    	dist = distance(x,y,pos[0],pos[1])
	    	if (dist < score):
	    		score = dist
	    		if (score < 20): #Instead check its within radius of nucleus
	    			label = key
	return label

def parse(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def reorganize(rootDir, depth):
	count = 0
	dirList = []
	for dirName, subdirList, fileList in walklevel(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		if dirName != rootDir:
			create(dirName)
			for fname in fileList:
				if (fname != ".DS_Store"):
					scanID = str(dirList.index(ntpath.basename(dirName))+1)
					nucID = str(int(parse(fname, "_", "_")))
					label = match(scanID, nucID)
					if label != "???":
						shutil.move(os.path.join(dirName,fname), os.path.join(dirName, label, fname))
					#delete the unknowns?
		count += 1
		if count > depth:
			break;

#Plots the coordinates of nuclei
#image is the name of the corresponding folder in nuclei folder (001, 002, 003, etc.)
def plot(scanDir, imageDir, scanID):
	xcoords = []
	ycoords = []
	image_path = glob.glob(os.path.join(scanDir, "*.psd"))[scanID-1]
	fileList = glob.glob(os.path.join(imageDir,"*.tiff"))
	for fname in fileList:
		nucID = str(int(parse(fname, "_", "_")))
		label = match(str(scanID), nucID)
		coord = coordinates(str(scanID), nucID)
		xcoords.append(coord[0])
		ycoords.append(coord[1])

	im = plt.imread(image_path)
	implot = plt.imshow(im)
	plt.scatter(xcoords,ycoords,s=4,c='black') #nuclei

	data_pos_dict = ppt.getPositions(image_path)
	choice1 = data_pos_dict[list(data_pos_dict.keys())[0]] # In python 2.7 this was choice1 = data_pos_dict[data_pos_dict.keys()[0]]
	x1,y1 = [i[0] for i in choice1],[i[1] for i in choice1]
	choice2 = data_pos_dict[list(data_pos_dict.keys())[1]]
	x2,y2 = [i[0] for i in choice2],[i[1] for i in choice2]
	choice3 = data_pos_dict[list(data_pos_dict.keys())[2]]
	x3,y3 = [i[0] for i in choice3],[i[1] for i in choice3]
	choice4 = data_pos_dict[list(data_pos_dict.keys())[3]]
	x4,y4 = [i[0] for i in choice4],[i[1] for i in choice4]

	plt.scatter(x1,y1,s=4,c='red') #micronuclei
	plt.scatter(x2,y2,s=4,c='cyan') #acute scar
	plt.scatter(x3,y3,s=4,c='violet') #subacute scar
	plt.scatter(x4,y4,s=4) #unscarred

	plt.show()


if __name__ == "__main__":
	reorganize('../output/nuclei', 3)
	plot("../psd/", "../output/nuclei/003", 3)
	#plot("../psd/", "../output/nuclei/002", 2)
	#plot("../psd/", "../output/nuclei/001/Micronuclei", 1)




