import infrastructure as inf

__author__ = 'Jakub Klapacz <jklapac2@illinois.edu>'

def main():

	all_arrays = inf.traverse_files()
	for entry in all_arrays:
		print entry





if __name__ == '__main__':
	main()