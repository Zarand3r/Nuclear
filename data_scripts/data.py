import make_input_data as make
import generate_nuclei_csv as generate
import optimize_data as optimize

SOURCE_DIRECTORY = "../../output/nuclei" 
OUTPUT_DIRECTORY = "../ImageData"

if __name__ == "__main__":

	make.write_data(SOURCE_DIRECTORY, OUTPUT_DIRECTORY)
	make.make_label_file()
	# optimize.optimize(OUTPUT_DIRECTORY, 0.7)
	generate.generate(OUTPUT_DIRECTORY, "nuclei.csv") #add filter parameters optional
	#There is also a function to plot all the labeled nuclei in a specific folder against the coordinates of all possible labels
