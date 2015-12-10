#!/usr/bin/env python

__author__ = "Henry Lin"

# Some numpy exercises. You could find some more here:
# http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
from numpy.testing import assert_equal

# Create a numpy array containing the digits 0 through 11
expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# array = None # TODO
array = range(0, 12)
array = np.arange(0, 12)
array = np.arange(12)
assert_array_equal(expected, array)

# Rearrange your numpy array, so it looks like..
expected = [[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11]]
# reshaped = None # TODO
reshaped = array.reshape(4, 3)
assert_array_equal(expected, reshaped)

print "PASSED THE RESHAPE TEST"

############# Attribute operations #############

# Get the dimensions (shape) of this array
expected = (4, 3)
shape = reshaped.shape
assert_equal(expected, shape)

# Get the number of columns of this array
expected = 3
num_cols = reshaped.shape[1]
assert_equal(expected, num_cols)

print "PASSED THE ATTRIBUTE TEST"

############# Access operations ###############

array = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8],
                  [9, 10, 11]])

# Grab element 10
expected = 10
element = array[3, 1] # TODO
# Compare to regular nested list you would do [3][1] (regular python)
assert_equal(expected, element)

# Extract the first row of your array
expected = [0, 1, 2]
first_row = array[0] # TODO
assert_array_equal(expected, first_row)

# Extract the last row of your array
expected = [9, 10, 11]
last_row = array[-1] # TODO
assert_array_equal(expected, last_row)

print "PASSED THE ACCESS TEST, PART 1"

# Extract the first column of your array - using "colon" notation
# NOTE: NEW SYNTAX
expected = [0, 3, 6, 9]
first_row = array[:, 0]
# Get every member of the first axis and only the 0-th element of the second dimension
assert_array_equal(expected, first_row)

# Extract the second column of your array
expected = [1, 4, 7, 10]
col_2 = array[:, 1] # TODO
assert_array_equal(expected, col_2)

print "PASSED THE ACCESS TEST, PART 2"

############# Slicing ###############
# You could look at even more crazy slice patterns here
# http://www.tp.umu.se/~nylen/pylect/intro/numpy/numpy.html#indexing-and-slicing

# Create an array that looks like this:
expected = [[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35]]
array = np.arange(36).reshape(6, 6)
assert_array_equal(expected, array)

# Grab the first three rows
expected = [[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17]]
slice1 = array[:3]  # Can also do array[0:3], last index is excluded
assert_array_equal(expected, slice1)

# Grab the middle two rows
expected = [[12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]]
slice2 = array[2:4] # TODO
assert_array_equal(expected, slice2)

# Grab the last four rows
expected = [[12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35]]
slice3 = array[-4:] # TODO
assert_array_equal(expected, slice3)

print "PASSED THE SLICING TEST, PART 1"

# Reminder
array = np.array([[ 0,  1,  2,  3,  4,  5],
                  [ 6,  7,  8,  9, 10, 11],
                  [12, 13, 14, 15, 16, 17],
                  [18, 19, 20, 21, 22, 23],
                  [24, 25, 26, 27, 28, 29],
                  [30, 31, 32, 33, 34, 35]])

# Grab the middle 2 columns
expected = [[ 2,  3],
            [ 8,  9],
            [14, 15],
            [20, 21],
            [26, 27],
            [32, 33]]
twocols = array[:, 2:4] # TODO
assert_array_equal(expected, twocols)

print "PASSED THE SLICING TEST, PART 2"

############# MATRIX AGGREGATE OPERATIONS ##########

# Create a matrix of shape (10, 3) of ones
expected = [[ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.],
            [ 1.,  1.,  1.]]
array = np.ones((10, 3))
assert_array_equal(expected, expected)

# Set the first column to 9
expected = [[ 9.,  1.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  1.,  1.]]
array[:, 0] = 9
assert_array_equal(expected, expected)

# Replace the second column with 0, 1, 2, ..., 9
expected = [[ 9.,  0.,  1.],
            [ 9.,  1.,  1.],
            [ 9.,  2.,  1.],
            [ 9.,  3.,  1.],
            [ 9.,  4.,  1.],
            [ 9.,  5.,  1.],
            [ 9.,  6.,  1.],
            [ 9.,  7.,  1.],
            [ 9.,  8.,  1.],
            [ 9.,  9.,  1.]]
# TODO
array[:, 1] = np.arange(10)
assert_array_equal(expected, array)

print "PASSED THE AGGREGATE OPERATIONS TEST, PART 1"

# Create a 1D array of consecutive numbers, from 1 through 10
expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
array = np.arange(1, 11)
assert_array_equal(expected, expected)

# Create a boolean mask. If array[i] < 6, then mask[i] = True
expected = [ True, True, True, True, True, False, False, False, False, False]
mask = (array < 6)
assert_array_equal(expected, mask)

# Set all elements less than 6 to zero in array
expected = [0, 0, 0, 0, 0, 6, 7, 8, 9, 10]
array[array < 6] = 0  # Can alternatively pass mask here
array[mask] = 0
assert_array_equal(expected, array)

array = np.arange(1, 11)

# Create a boolean mask. If array[i] is even, then mask[i] = True
expected = [False,  True, False,  True, False,  True, False,  True, False,  True]
mask = (array % 2 == 0) # TODO
array[mask]
assert_array_equal(expected, mask)

# Set all elements in array that are even to 888
expected = [  1, 888,   3, 888,   5, 888,   7, 888,   9, 888]
# TODO
array[mask] = 888
assert_array_equal(expected, array)

print "PASSED THE AGGREGATE OPERATIONS TEST, PART 2"

array = np.array([[ 9.,  0.,  1.],
                  [ 9.,  1.,  1.],
                  [ 9.,  2.,  1.],
                  [ 9.,  3.,  1.],
                  [ 9.,  4.,  1.],
                  [ 9.,  5.,  1.],
                  [ 9.,  6.,  1.],
                  [ 9.,  7.,  1.],
                  [ 9.,  8.,  1.],
                  [ 9.,  9.,  1.]])

# Assign an element a_i2 (an element in the third column) as 666 if
# a_i1 is an even number
expected = [[ 9.,  0.,  666.],
            [ 9.,  1.,  1.],
            [ 9.,  2.,  666.],
            [ 9.,  3.,  1.],
            [ 9.,  4.,  666.],
            [ 9.,  5.,  1.],
            [ 9.,  6.,  666.],
            [ 9.,  7.,  1.],
            [ 9.,  8.,  666.],
            [ 9.,  9.,  1.]]
# TODO
array[array[:, 1] % 2 == 0, 2] = 666
assert_array_equal(expected, array)

print "PASSED THE AGGREGATE OPERATIONS TEST, PART 3"

############# MATRIX MULTIPLCATION ##########

vector1 = np.ones(10) * 2
vector2 = np.arange(10)

# Perform vector1 "dot" vector2
expected = 90
product = vector1.dot(vector2) # TODO np.dot(vector1, vector2)
assert_equal(expected, product)

# Perform the elementwise multiplaction of vector1 and vector2
expected = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
product = vector1 * vector2
assert_array_equal(expected, product)
matrix = np.arange(60).reshape(10, 6)

# Matrix operations: product "dot" matrix
expected = [ 3420.,  3510.,  3600.,  3690.,  3780.,  3870.]
result = product.dot(matrix) # TODO
assert_array_equal(expected, result)

print "PASSED THE PRODUCT TEST"