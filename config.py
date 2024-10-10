import os
import sys
import yaml

os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["GST_DEBUG"] = "1"

with open('env.yml') as f:
	env = yaml.load(f, Loader=yaml.FullLoader)
	# import customed compiled cv2
	if not env["opencv_path"]:
		print("OpenCV path is not set in env.yml") # import cv2 -> print(cv2.__info__) -> take the path where the cv2 is located
		sys.exit(1)
	sys.path.insert(0, env["opencv_path"])
	import cv2
	del sys.path[0]

with open('apikey.yml') as f:
	apikey = yaml.load(f, Loader=yaml.FullLoader)
