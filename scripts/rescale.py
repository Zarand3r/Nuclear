import numpy as np
import os
import sys
from PIL import Image
from PIL import ImageDraw
import random
import ntpath

# def walklevel(some_dir, level=1):
#     some_dir = some_dir.rstrip(os.path.sep)
#     assert os.path.isdir(some_dir)
#     num_sep = some_dir.count(os.path.sep)
#     for root, dirs, files in os.walk(some_dir):
#         yield root, dirs, files
#         num_sep_this = root.count(os.path.sep)
#         if num_sep + level <= num_sep_this:
#             del dirs[:]

def padding(rootDir, depth, dimx, dimy, exclude = [0]):
	count = 0
	dirList = []
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		else:
			if ntpath.basename(dirName) in dirList:
				count +=1
				if count > depth:
					break;
			if count not in exclude:
				for fname in fileList:
					if (fname != ".DS_Store"):
						img1 = Image.open(os.path.join(dirName, fname))
						width, height = img1.size
						padx = int((dimx-width)/2)
						pady = int((dimy-height)/2)
						#account for case with odd padding, splitting the extra pixel row/column randomly 
						dx1 = 0
						dx2 = 0 
						dy1 = 0
						dy2 = 0
						if padx*2 < (dimx-width):
							dx1 = 1
							if (bool(random.getrandbits(1))):
								dx1 = 0
								dx2 = 1
						if pady*2 < (dimy-height):
							dy1 = 1
							if (bool(random.getrandbits(1))):
								dy1 = 0
								dy2 = 1
						img2 = img1.crop((-dx1+dx2-padx,-dy1+dy2-pady,(dimx-dx1+dx2)-padx,(dimy-dy1+dy2)-pady))  
						draw = ImageDraw.Draw(img2)
						draw.rectangle((0,0,dimx,(pady+dy1)), fill="black" )
						draw.rectangle((0,dimy-(pady+dy2),dimx,dimy), fill="black" )
						draw.rectangle((0,0,(padx+dx1),dimy), fill="black" )
						draw.rectangle((dimx-(padx+dx2),0,dimx,dimy), fill="black" )
						del draw
						img2.save(os.path.join(dirName, fname), "tiff") #there is a quality=75 parameter i deleted
						print('\t%s' % fname)


def revert(rootDir, depth):
	count = 0
	dirList = []
	for dirName, subdirList, fileList in os.walk(rootDir):
		print('Found directory: %s' % dirName)
		if dirName == rootDir:
			dirList = subdirList
		else:
			for fname in fileList:
				if (fname != ".DS_Store"):
					img = Image.open(os.path.join(dirName, fname))
					w, h = img.size
					cropped_img = img.crop((w//2 - 52//2, h//2 - 52//2, w//2 + 52//2, h//2 + 52//2))
					cropped_img.save(os.path.join(dirName, fname), "tiff")
			if ntpath.basename(dirName) in dirList:
				count +=1
				if count > depth:
					break;

if __name__ == "__main__":
	# padding('../output/nuclei', 5, 54, 54)
	revert('../output/nuclei', 1)	





