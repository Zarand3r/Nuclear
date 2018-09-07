import os
import ntpath
import csv
import random 

# DATA_DIRECTORY = "../output/nuclei"
DATA_DIRECTORY = "../ImageData"
LABELS = ["Acute_Scar", "Micronuclei", "Subacute_Scar", "Unscarred"]

def filter(dirName, ratio):
	index = LABELS.index(ntpath.basename(dirName))
	if index >= 0:
		if random.random() < ratio[index]:
			return True
	return False


def generate(rootDir, output_file, ratio = [1, 1, 1, 1]):
	data = open(output_file, 'w')
	writer = csv.writer(data)
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if ntpath.basename(dirName) in LABELS:
			for fname in fileList:
				if fname != ".DS_Store":
					if (filter(dirName, ratio)):
						writer.writerow([os.path.join(dirName, fname), ntpath.basename(dirName)])

if __name__ == "__main__":
	# generate(DATA_DIRECTORY, "nuclei.csv")
	# generate(DATA_DIRECTORY, "nuclei.csv", [0.5, 0.5, 0.5, 0.1])
	generate(DATA_DIRECTORY, "nuclei.csv", [1, 1, 1, 0.5])