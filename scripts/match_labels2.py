import numpy as np
import os
import sys
import random
import psd_parse_tools2 as ppt
import math
import ntpath
import csv
import matplotlib.pyplot as plt
import glob
import shutil

LABELS = ["Acute_Scar", "Micronuclei", "Subacute_Scar", "Unscarred"]

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

#Temp code to run in the reorganize function. This rerases the folders with wrong names
def delete(rootDir):
	dirList = []
	cwd = os.getcwd()
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		if ntpath.basename(dirName) in dirList:
			os.chdir(dirName)
			if os.path.isdir("Acute Scar"):
				os.rename("Acute Scar", "Acute_Scar")
			if os.path.isdir("Subacute Scar"):
				os.rename("Subacute Scar", "Subacute_Scar")
			os.chdir(cwd)


#Creates 4 subfolders (for each nuclear morphology label) in each nuclei folder. 
def create(dirName):
	for label in LABELS:
		os.makedirs(os.path.join(dirName, label), exist_ok=True) 

#Calculates distance between two coordinates
def distance(x1,y1,x2,y2):
    sq1 = (x1-x2)*(x1-x2)
    sq2 = (y1-y2)*(y1-y2)
    return math.sqrt(sq1 + sq2)

def coordinates(scanID, nucID):
	csv_file = open('../output/Measurements/Nucleus_BAF_Nuclei.csv', "r")
	reader = csv.reader(csv_file, delimiter=",")
	for row in reader:
		if (str(scanID) == row[0] and str(nucID) == row[1]):
			return (float(row[3]), float(row[4]), float(row[13]))
	csv_file.close()

def match(fname, dirName, dirList, label_list, label_positions):
	scanID = dirList.index(ntpath.basename(dirName))+1
	nucID = int(parse(fname, "_", "_"))
	(x,y,r) = coordinates(scanID, nucID)
	score = 100
	label = "???"
	for (key,value) in label_positions.items():
	    for index, pos in enumerate(value):
	    	dist = distance(x,y,pos[0],pos[1])
	    	if (dist < score):
	    		score = dist
	    		if (score < 10+r): 
	    			if (key == "Micronuclei" and r>5):
	    				continue 
	    			else:
		    			if (pos[2] == 0):
		    				label = key
		    				value[index][2] = nucID
		    				value[index][3] = score
		    			else:
		    				if (score < pos[3]):
		    					label = key
		    					label_list[pos[2]-1] = (label_list[pos[2]-1][0], label_list[pos[2]-1][1], "???")
		    					value[index][2] = nucID
		    					value[index][3] = score
	label_list.append((dirName, fname, label))


def parse(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def organize(rootDir, depth, exclude = [0]):
	count = 0
	dirList = []
	for dirName, subdirList, fileList in walklevel(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		if count not in exclude: #for testing purposes. Otherwise just if dirName != rootDir
			delete(dirName)
			create(dirName)
			label_list = []
			scanID = str(dirList.index(ntpath.basename(dirName))+1)
			image_path = glob.glob(os.path.join("../psd/", "*.psd"))[int(scanID)-1]
			label_positions = ppt.getPositions(image_path)

			for fname in fileList:
				if (fname != ".DS_Store"):
					match(fname, dirName, dirList, label_list, label_positions)
			for (dirName, fname, label) in label_list:
				if label == "Acute Scar":
					label = "Acute_Scar"
				if label == "Subacute Scar":
					label = "Subacute_Scar"
				if label in LABELS:
					shutil.move(os.path.join(dirName,fname), os.path.join(dirName, label, fname))
		count += 1
		if count > depth:
			break;



def reset(rootDir, depth):
	count = 0
	nucDir = rootDir
	dirList = []
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		else:
			if ntpath.basename(dirName) in dirList:
				nucDir = dirName
				count += 1
				if count > depth:
					break;
			for fname in fileList:
				if (fname != ".DS_Store"):
					shutil.move(os.path.join(dirName, fname), os.path.join(nucDir,fname))

def resetDir(nucDir):
	for dirName, subdirList, fileList in os.walk(nucDir):
		for fname in fileList:
			if (fname != ".DS_Store"):
				shutil.move(os.path.join(dirName, fname), os.path.join(nucDir,fname))


#Plots the coordinates of nuclei
#image is the name of the corresponding folder in nuclei folder (001, 002, 003, etc.)
def plot(scanDir, imageDir, scanID):
	xcoords = []
	ycoords = []
	image_path = glob.glob(os.path.join(scanDir, "*.psd"))[scanID-1]
	fileList = glob.glob(os.path.join(imageDir,"*.tiff"))
	data_pos_dict = ppt.getPositions(image_path)

	for fname in fileList:
		nucID = int(parse(ntpath.basename(fname), "_", "_"))
		coord = coordinates(scanID, nucID)
		xcoords.append(coord[0])
		ycoords.append(coord[1])

	im = plt.imread(image_path)
	implot = plt.imshow(im)
	plt.scatter(xcoords,ycoords,s=4,c='black') #nuclei

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
	# organize('../output/nuclei', 5, [0, 1, 2])
	plot("../psd/", "../output/nuclei/Case_214_L10_40/Subacute_Scar", 9)
	#plot("../psd/", "../output/nuclei/003", 3)
	#plot("../psd/", "../output/nuclei/002", 2)
	#plot("../psd/", "../output/nuclei/001/Micronuclei", 1)




