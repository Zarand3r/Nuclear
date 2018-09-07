# image_clustering
2d clustering of cell histology data
requires python packages: numpy,sys,math,copy,matplotlib,scipy

```
pip install numpy sys math copy matplotlib scipy
```

run as:
```
./construct-graph-new.py ARGV[1] ARGV[2]
```
where:

**ARGV[1]** is path to a image file: png,jpeg

**ARGV[2]**	is a float between 0 and 1 which defines a image instensity cutoff

this python script works on images 1000x1000 pixels in size
