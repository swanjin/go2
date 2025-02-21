#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
import config
import robot_dog
from ui import RobotDogUI
import threading

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide font
    app.setFont(QFont('Arial', 10))
    
    mydog = robot_dog.Dog(config.env, config.apikey)
    mydog.setup()

    mydog.tts_finished_event = threading.Event()
    
    window = RobotDogUI(mydog)
    window.show()
    
    sys.exit(app.exec())
