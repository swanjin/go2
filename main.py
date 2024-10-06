#!/usr/bin/env python3
import robot_dog
import os
import yaml

if __name__ == "__main__":
    os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
    os.environ["GST_DEBUG"] = "1"

    with open('env.yml') as f:
        env = yaml.load(f, Loader=yaml.FullLoader)
    with open('apikey.yml') as f:
        apikey = yaml.load(f, Loader=yaml.FullLoader)

    mydog = robot_dog.Dog(env, apikey)
    mydog.setup()
    mydog.run_gpt()
    mydog.shutdown()
