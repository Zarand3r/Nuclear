import pandas as pd
from scipy import misc
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
import csv

#generate nuclei csv first 

samples = []
INPUT = "nuclei2.csv"
num_images = 0
colors = []

nuclei = open(INPUT, "r")
reader = csv.reader(nuclei, delimiter=",")
for row in reader:
	num_images += 1
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
	# the following line changed
	samples.append(img.reshape(-1))
nuclei.close()

df = pd.DataFrame.from_records(samples, coerce_float=True)

# print(df.head())


iso = manifold.Isomap(n_neighbors=6, n_components=2)
iso.fit(df)

manifold_2Da = iso.transform(df)
manifold_2D = pd.DataFrame(manifold_2Da, columns=['Component 1', 'Component 2'])

# Left with 2 dimensions
manifold_2D.head()

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot(111)
ax.set_title('2D Components from Isomap of Facial Images')
ax.set_xlabel('Component: 1')
ax.set_ylabel('Component: 2')

# Show 40 of the images ont the plot
x_size = (max(manifold_2D['Component 1']) - min(manifold_2D['Component 1'])) * 0.08
y_size = (max(manifold_2D['Component 2']) - min(manifold_2D['Component 2'])) * 0.08
for i in range(40):
	img_num = np.random.randint(0, num_images)
	if colors[img_num] == "red":
		x0 = manifold_2D.loc[img_num, 'Component 1'] - (x_size / 2.)
		y0 = manifold_2D.loc[img_num, 'Component 2'] - (y_size / 2.)
		x1 = manifold_2D.loc[img_num, 'Component 1'] + (x_size / 2.)
		y1 = manifold_2D.loc[img_num, 'Component 2'] + (y_size / 2.)
		img = df.iloc[img_num,:].values.reshape(52, 52)
		ax.imshow(img, aspect='auto', cmap=plt.cm.gray, 
				  interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

# Show 2D components plot
ax.scatter(manifold_2D['Component 1'], manifold_2D['Component 2'], marker='.', c = colors, alpha=0.7)

ax.set_ylabel('Up-Down Pose')
ax.set_xlabel('Right-Left Pose')

plt.show()
