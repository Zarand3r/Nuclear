import os
import sys
import random
import ntpath
import glob
import shutil
import csv

#code to rename folders in nuclei by index (001, 002, 003, etc)
#code to rename nuclei files with case info and xy info
def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def coordinates(scanID, nucID):
	csv_file = open('../output/Measurements/Nucleus_BAF_Nuclei.csv', "r")
	reader = csv.reader(csv_file, delimiter=",")
	for row in reader:
		if (str(scanID) == row[0] and str(nucID) == row[1]):
			return (int(float(row[3])), int(float(row[4])))
	csv_file.close()

def parse(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def rename_scans(rootDir, depth, byindex = False):
	count = 0
	dirList = []
	for dirName, subdirList, fileList in walklevel(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		elif ntpath.basename(dirName) in dirList:
			scanID = dirList.index(ntpath.basename(dirName))
			if (byindex):
				scanName = str("00"+str(scanID + 1))
			else:
				image_path = glob.glob(os.path.join("../psd/", "*.psd"))[scanID]
				scanName = os.path.splitext(ntpath.basename(image_path))[0]
			print(scanName)
			shutil.move(dirName, os.path.join(rootDir, scanName))			
		count += 1
		if count > depth:
			break;

def rename_nuclei(rootDir, depth, exclude = [0]):
	scanID = 0
	dirList = []
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		else:
			if ntpath.basename(dirName) in dirList:
				scanID = dirList.index(ntpath.basename(dirName))+1
				if scanID > depth:
					break;
			if scanID not in exclude:
				for fname in fileList:
					if fname != ".DS_Store":
						nucindex = parse(fname, "_", "_")
						nucID = str(int(nucindex))
						x,y = coordinates(scanID, nucID)
						newfilename = "Nucleus"+"_"+nucindex+"_"+str(x)+"_"+str(y)+".tiff"
						shutil.move(os.path.join(dirName, fname), os.path.join(dirName,newfilename))

def add_zeroes(rootDir, depth):
	scanID = 1
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		elif ntpath.basename(dirName) in dirList:
			scanID = dirList.index(ntpath.basename(dirName))+1
			if scanID > depth:
				break;
		for fname in fileList:
			if fname != ".DS_Store":
				nucindex = parse(fname, "_", "_")
				newindex = nucindex
				nucID = int(nucindex)
				if nucID < 10:
					newindex = "000"+str(nucID)
				elif nucID < 100:
					newindex = "00"+str(nucID)
				elif nucID < 1000:
					newindex = "0"+str(nucID)
				newfilename = fname.replace("_"+nucindex+"_", "_"+newindex+"_")
				shutil.move(os.path.join(dirName, fname), os.path.join(dirName, newfilename))


#Make a function to add 00 or 000 in front of the nuclei number 

if __name__ == "__main__":
	# rename_scans('../output/nuclei', 66)
	# rename_nuclei('../output/nuclei', 5)
	# rename_nuclei('../output/crops', 10)
	add_zeroes('../output/nuclei', 7)
