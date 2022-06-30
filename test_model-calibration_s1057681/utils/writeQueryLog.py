
import numpy as np
import os, glob, time, datetime

# add safeguards

if not os.path.exists('query.log'): # if not exist then write
	f = open('query.log', 'w')
	currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	f.write('Start querying at\n')
	f.write('%s\n' % currentTime)
	f.close()
else:
	f = open('query.log')
	txt = f.readlines()
	f.close()
	if len(txt) == 2: # if there are only 2 lines
		f = open('query.log', 'a')
		currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		f.write('Stop querying at\n')
		f.write('%s\n' % currentTime)
		f.close()
	else: # if not 2 lines, then rewrite
		f = open('query.log', 'w')
		currentTime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		f.write('Start querying at\n')
		f.write('%s\n' % currentTime)
		f.close()		
