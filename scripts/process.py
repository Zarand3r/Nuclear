import rename as rename
import rescale as rescale 
import match_labels2 as label

NUCLEI_DIRECTORY = "../output/nuclei"
DEPTH = 25
EXCLUDE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

if __name__ == "__main__":
	# rename.rename_scans(NUCLEI_DIRECTORY, DEPTH)
	# rename.add_zeroes(NUCLEI_DIRECTORY, DEPTH)
	rename.rename_nuclei(NUCLEI_DIRECTORY, DEPTH, EXCLUDE)
	rescale.padding(NUCLEI_DIRECTORY, DEPTH, 52, 52, EXCLUDE)
	# label.reset(NUCLEI_DIRECTORY, DEPTH)
	label.organize(NUCLEI_DIRECTORY, 23, EXCLUDE) 

	#There is also a function to plot all the labeled nuclei in a specific folder against the coordinates of all possible labels
