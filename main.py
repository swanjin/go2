#!/usr/bin/env python3
import config
import robot_dog

if __name__ == "__main__":
    mydog = robot_dog.Dog(config.env, config.apikey)
    mydog.setup()
    mydog.run_gpt()
    mydog.shutdown()
