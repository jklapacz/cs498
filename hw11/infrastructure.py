import os
import numpy as np
from collections import defaultdict

def traverse_files():
	all_arrays = defaultdict(list)
	for dirname in os.listdir("./HMP_Dataset"):
		# print dirname
		# continue
		# if not os.path.isdir(dirname): 
		# 	print dirname + 'is not a directory'
		# 	continue
		for filename in os.listdir(dirname):
			if filename.startswith("."): continue  # Edit added 12/7
			print "Loading in: ", dirname + os.sep + filename
			print dirname
			array = np.loadtxt(dirname + os.sep + filename)
	 		
	 		all_arrays[dirname].append(array)
	 
	return all_arrays