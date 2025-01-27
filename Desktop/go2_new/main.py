#!/usr/bin/env python3
import sys
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont
import config
import robot_dog
from ui import RobotDogUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide font
    app.setFont(QFont('Arial', 10))
    
    mydog = robot_dog.Dog(config.env, config.apikey)
    mydog.setup()
    
    window = RobotDogUI(mydog)
    window.show()
    
    sys.exit(app.exec())


#!/usr/bin/env python3

# import config
# import robot_dog

# if __name__ == "__main__":
#     mydog = robot_dog.Dog(config.env, config.apikey)
#     mydog.setup()
#     mydog.run_gpt()
#     mydog.shutdown()