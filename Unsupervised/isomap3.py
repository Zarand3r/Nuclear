import pandas as pd
from scipy import misc
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import csv
import numpy as np
import math


#generate nuclei csv first 

samples = []
IMAGES = "nuclei.csv"
FEATURES = "reduced.csv"
colors = []
discard = []

def Optimize(coordinates):
	first = colors.index("red")
	second = colors.index("cyan")
	third = colors.index("orange")
	fourth = colors.index("blue")
	acute = np.asarray(coordinates[first: second])
	micro = np.asarray(coordinates[second: third])
	sub = np.asarray(coordinates[third: fourth])
	unscar = np.asarray(coordinates[fourth:])

	acutemean = np.mean(acute, axis=0)
	acutestd = np.std(acute, axis=0)
	micromean = np.mean(micro, axis=0)
	microstd = np.std(micro, axis=0)
	submean = np.mean(sub, axis=0)
	substd = np.std(sub, axis=0)
	unscarmean = np.mean(unscar, axis=0)
	unscarstd = np.std(unscar, axis=0)


	for index, point in enumerate(coordinates):
		if (index in range(first, second)):
			if abs(point[1] - acutemean[1]) > acutestd[1]:
				discard.append(index)

		if (index in range(second, third)):
			if abs(point[1] - micromean[1]) > microstd[1]:
				discard.append(index)

		if (index in range(third, fourth)):
			if abs(point[1] - submean[1]) > substd[1]:
				discard.append(index)

		if (index in range(fourth, len(coordinates))):
			if abs(point[1] - unscarmean[1]) > unscarstd[1]:
				discard.append(index)

	for i in range(len(discard)-1, -1, -1):
		coordinates = np.delete(coordinates, discard[i], 0)
		del colors[discard[i]]

	return coordinates



def Plot2D(T, title, x, y):
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_title(title)
	ax.set_xlabel('Component: {0}'.format(x))
	ax.set_ylabel('Component: {0}'.format(y))
	x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
	y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
	# It also plots the full scatter:
	ax.scatter(T[:,x],T[:,y], marker='.', c = colors, alpha=0.7)


#add colors!!

def Plot3D(T, title, x, y, z):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection = '3d')
	ax.set_title(title)
	ax.set_xlabel('Component: {0}'.format(x))
	ax.set_ylabel('Component: {0}'.format(y))
	ax.set_zlabel('Component: {0}'.format(z))
	x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
	y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
	z_size = (max(T[:,z]) - min(T[:,z])) * 0.08
	# It also plots the full scatter:
	ax.scatter(T[:,x],T[:,y],T[:,z], marker='.', c = colors, alpha=0.65)

	# Show 40 of the images ont the plot

def plotImageData():
	nuclei = open(IMAGES, "r")
	reader = csv.reader(nuclei, delimiter=",")
	for row in reader:
		# if "(" not in row[0]:
		# 	label = row[1]
		# 	color = "blue"
		# 	if label == "Acute_Scar":
		# 		color = "red"
		# 	elif label == "Micronuclei":
		# 		color = "cyan"
		# 	elif label == "Subacute_Scar":
		# 		color = "orange"
		# 	colors.append(color)
		# 	img = misc.imread(row[0], mode='I')
		# 	samples.append(img.reshape(-1))
		label = row[1]
		color = "blue"
		if label == "Acute_Scar":
			color = "red"
		elif label == "Micronuclei":
			color = "cyan"
		elif label == "Subacute_Scar":
			color = "orange"
		colors.append(color)
		img = misc.imread(row[0], mode='I')
		samples.append(img.reshape(-1))
	nuclei.close()

	df = pd.DataFrame.from_records(samples, coerce_float=True)
	iso = manifold.Isomap(n_neighbors = 6, n_components = 3)
	Z = iso.fit_transform(df)
	Z = Optimize(Z)
	Plot2D(Z, "Isomap transformed data, 2D", 0, 1)
	Plot3D(Z, "Isomap transformed data 3D", 0, 1, 2)
	plt.show()

def plotFeatureData():
	nuclei = open(FEATURES, "r")
	reader = csv.reader(nuclei, delimiter=",")
	index = 0
	for row in reader:
		if index != 0:
			# data_array = np.asarray([row[2], row[6], row[7], row[9], row[11], row[13], row[17], row[53], row[56], row[57], row[61], row[62], row[65]])
			data_array = np.asarray([row[2], row[6], row[7], row[9], row[10], row[11], row[12], row[13], row[14], row[16], row[17], row[19], row[20], row[53], row[56], row[57], row[61], row[62], row[65]])
			samples.append(data_array.reshape(-1))
		index += 1
	nuclei.close()

	df = pd.DataFrame.from_records(samples, coerce_float=True)
	iso = manifold.Isomap(n_neighbors = 6, n_components = 3)
	Z = iso.fit_transform(df)

	Plot2D(Z, "Isomap transformed data, 2D", 0, 1)
	Plot3D(Z, "Isomap transformed data 3D", 0, 1, 2)
	plt.show()

if __name__ == "__main__":
	# plotFeatureData()
	plotImageData()


