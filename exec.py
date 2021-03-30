import os, time

for i in range(0,10):
	os.system('python run.py adult')
	time.sleep(10)
	os.system('python run.py compas')
	time.sleep(10)
	os.system('python run.py hospital')
	time.sleep(10)