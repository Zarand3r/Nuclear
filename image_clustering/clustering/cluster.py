#!/usr/bin/env python

import sys,math,copy
import numpy as np
from scipy import linalg as LA

class cluster:
	def __init__(self,cn):
		self.cluster_num = cn
		self.n = 0
		self.indiciesx = []
		self.indiciesy = []

	def add_index(self,i,j):
		self.indiciesx.append(i)
		self.indiciesy.append(j)
		self.n += 1

	def geometry_calculations(self):
		self.xcent = sum(self.indiciesx)/self.n 
		self.ycent = sum(self.indiciesy)/self.n
		dx = [(self.indiciesx[i] - self.xcent) for i in range(self.n)]
		dy = [(self.indiciesy[i] - self.ycent) for i in range(self.n)]
		self.Rg = sum([ dx[i]**2 + dy[i]**2 for i in range(self.n)])/self.n
		xx2 = sum([ dx[i]*dx[i] for i in range(self.n)])/self.n
		xy2 = sum([ dx[i]*dy[i] for i in range(self.n)])/self.n
		yy2 = sum([ dy[i]*dy[i] for i in range(self.n)])/self.n
		Rg_matrix = np.array([[xx2,xy2],[xy2,yy2]])
		e_vals,e_vec = LA.eig(Rg_matrix)
		self.relative_shape = (3/2)*(e_vals[0].real**4 + e_vals[1].real**4)/((e_vals[0].real**2 + e_vals[1].real**2)**2) - (1/2)

def load_png_image_file(fid,channel):
	img = mpimg.imread(path_to_file)
	org_img = copy.deepcopy(img)
	lum_img = img[:,:,channel]  #chooseing one of three img[:,:,1] img[:,:,2]
	len_lum = np.shape(lum_img)	
	return lum_img,len_lum,org_img

def get_near_list(i,pixel_region,mlen):
	irange = list(range(i-pixel_region,i+pixel_region+1,1)); to_pop = [];
	for i2 in irange:
		if i2 < 0 or i2 >= mlen:
			to_pop.append(i2) 				
	if len(to_pop) > 0:
		for i3 in to_pop:
			irange.remove(i3)
	return irange

def locate_clusters(lum_img,len_lum):
	pixel_region = 2;   group_id = 0
	mlenx = len_lum[0]; mleny = len_lum[1]
	group_img = np.zeros(shape = (mlenx,mleny))
	for i in range(mlenx):
		irangex = get_near_list(i,pixel_region,mlenx)
		for j in range(mleny):
			irangey = get_near_list(j,pixel_region,mleny)
			if lum_img[i][j] == 1.0:
				if group_img[i][j] == 0.0:
					group_id = group_id + 1
					group_img[i][j] = group_id
				for m in irangex:
					for n in irangey:
						if m == i and n == j:
							pass
						else:
							if lum_img[m][n] == 1.0:
								if group_img[m][n] == 0.0:
									group_img[m][n] = group_img[i][j] 
								elif group_img[m][n] != group_img[i][j]:  # seperate clusters become one
									to_remove = group_img[m][n]
									to_replace = group_img[i][j]
									for dex,valx in enumerate(group_img):
										for dey,valy in enumerate(valx):
											if valy == to_remove:
												group_img[dex][dey] = to_replace								
								else: 					 # add m,n to the cluster	
									group_img[m][n] = group_img[i][j]
	return group_img

def generate_clusters(group_img):
	unique_values = set(x for l in group_img for x in l)
	unique_values.discard(0.0)
	clust = []
	for el in unique_values:
		temp_clust = cluster(el)
		indicies = np.where(group_img == el)
		xval = [indicies[0][n] for n in range(len(indicies[1]))]
		yval = [indicies[1][n] for n in range(len(indicies[1]))]
		for j in range(len(xval)):
			temp_clust.add_index(xval[j],yval[j])
		clust.append(temp_clust)
	return clust

def cluster_cleanup(clust):
	ret_clust = []
	for clu in clust:
		if clu.n > 40:
			ret_clust.append(clu)
	return ret_clust
