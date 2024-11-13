from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit,
                           QScrollArea, QFrame)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette
import sys
import cv2
from PIL import Image
import numpy as np

class ChatMessage(QFrame):
    def __init__(self, text, is_user=False, image=None, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 5, 10, 5)
        
        # Message container
        container = QFrame()
        container.setMaximumWidth(600)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        
        # Sender label (Go2 or User)
        sender = QLabel("You" if is_user else "Go2")
        sender.setStyleSheet(f"""
            QLabel {{
                color: {'#1A73E8' if is_user else '#666666'};
                font-weight: bold;
                font-size: 13px;
            }}
        """)
        container_layout.addWidget(sender)
        
        # Message bubble
        message = QLabel(text)
        message.setWordWrap(True)
        message.setStyleSheet(f"""
            QLabel {{
                background-color: {'#1A73E8' if is_user else '#F1F3F4'};
                color: {'white' if is_user else 'black'};
                border-radius: 15px;
                padding: 10px 15px;
                font-size: 14px;
            }}
        """)
        container_layout.addWidget(message)
        
        # Add image if provided
        if image is not None:
            img_label = QLabel()
            pixmap = QPixmap.fromImage(image)
            scaled_pixmap = pixmap.scaled(
                400, 300, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            img_label.setPixmap(scaled_pixmap)
            img_label.setStyleSheet("""
                QLabel {
                    border-radius: 10px;
                    margin-top: 5px;
                }
            """)
            container_layout.addWidget(img_label)

        # Align messages
        main_layout.addStretch() if is_user else None
        main_layout.addWidget(container)
        main_layout.addStretch() if not is_user else None

class RobotDogUI(QMainWindow):
    def __init__(self, dog_instance):
        super().__init__()
        self.dog = dog_instance
        self.feedback_mode = False
        self.search_started = False
        self.target_set = False
        self.conversation_started = False
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Go2 Chat Interface')
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
            }
        """)

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Chat area with scroll
        self.chat_area = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_area)
        self.chat_layout.setSpacing(15)
        self.chat_layout.addStretch()

        # Start button
        self.start_button = QPushButton("Start Conversation")
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #1A73E8;
                color: white;
                border-radius: 20px;
                padding: 15px 30px;
                font-size: 16px;
                font-weight: bold;
                margin: 20px;
            }
            QPushButton:hover {
                background-color: #1557AA;
            }
        """)
        self.start_button.clicked.connect(self.start_conversation)
        self.chat_layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.chat_area)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
            }
        """)
        layout.addWidget(self.scroll)

        # Input area (initially hidden)
        self.input_widget = QWidget()
        self.input_widget.setStyleSheet("""
            QWidget {
                background-color: #F1F3F4;
                border-top: 1px solid #E0E0E0;
            }
        """)
        input_layout = QHBoxLayout(self.input_widget)
        input_layout.setContentsMargins(20, 20, 20, 20)
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Send a message...")
        self.message_input.setStyleSheet("""
            QLineEdit {
                border: 1px solid #E0E0E0;
                border-radius: 20px;
                padding: 10px 15px;
                background-color: white;
                color: #333333;
                font-size: 14px;
                min-height: 20px;
            }
            QLineEdit:focus {
                border: 1px solid #1A73E8;
                outline: none;
            }
            QLineEdit::placeholder {
                color: #999999;
            }
        """)
        self.message_input.returnPressed.connect(self.send_message)
        
        send_button = QPushButton("Send")
        send_button.setStyleSheet("""
            QPushButton {
                background-color: #1A73E8;
                color: white;
                border-radius: 20px;
                padding: 10px 20px;
                font-weight: bold;
                border: none;
            }
            QPushButton:hover {
                background-color: #1557AA;
            }
        """)
        send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(send_button)
        
        self.input_widget.hide()  # Initially hide the input area
        layout.addWidget(self.input_widget)

    def start_conversation(self):
        # Remove start button and show input area immediately
        self.start_button.hide()
        self.start_button.deleteLater()
        self.input_widget.show()
        
        # Add robot message immediately
        welcome_message = "Hello! I'm Go2, your robot dog assistant. What would you like me to find for you?"
        self.add_robot_message(welcome_message)
        
        # Set conversation started flag before TTS
        self.conversation_started = True
        self.message_input.setFocus()
        
        # Play TTS after UI updates
        if self.dog.env["tts"]:
            QTimer.singleShot(100, lambda: self.dog.ai_client.tts(welcome_message))

    def send_message(self):
        if not self.conversation_started:
            return
            
        text = self.message_input.text().strip()
        if not text:
            return
        
        # Always show user message first
        self.add_user_message(text)
        self.message_input.clear()

        # Status-related questions
        status_questions = [
            "what are you doing",
            "what's happening",
            "current status",
            "status",
            "what is happening",
            "what's going on",
            "how is it going",
            "found anything",
            "see anything"
        ]

        if any(q in text.lower() for q in status_questions):
            if self.target_set:
                status = f"I'm currently searching for {self.dog.target}. "
                if self.feedback_mode:
                    status += "I'm in feedback mode, waiting for your guidance."
                else:
                    status += "I'm in automatic search mode."
            else:
                status = "I'm waiting for you to tell me what to search for."
            
            # First show status message in UI
            self.add_robot_message(status)
            # Then play TTS after UI update
            # QTimer.singleShot(300, lambda: self.play_tts(status))
            
        elif text.lower() == "feedback":
            # First pause the search
            self.dog.feedback_complete_event.clear()
            self.dog.interrupt_round_flag.set()
            self.feedback_mode = True
            
            # Show feedback mode activation in UI
            feedback_msg = "Feedback mode activated. Please provide your feedback."
            self.add_robot_message(feedback_msg)
            
            # Play TTS after UI update
            if self.dog.env["tts"]:
                QTimer.singleShot(300, lambda: self.play_tts(feedback_msg))
            
        elif self.feedback_mode:
            # Process feedback with proper UI updates first
            assistant = self.dog.ai_client.get_response_by_feedback(text)
            if assistant:
                response_text = f"Action: {assistant.action}\nReason: {assistant.reason}"
                self.add_robot_message(response_text)
                
                # Execute action after UI update
                QTimer.singleShot(300, lambda: self.execute_feedback_action(assistant))
        elif not self.target_set:
            # First show the response in UI
            response = f"I'll start searching for {text} now."
            self.add_robot_message(response)
            
            # Then set target and start search after UI update
            QTimer.singleShot(100, lambda: self.process_target(text, response))
                
        else:
            self.dog.feedback = text

    def play_tts(self, message):
        """Helper method to play TTS"""
        if self.dog.env["tts"]:
            self.dog.ai_client.tts(message)

    def process_target(self, text, response):
        """Helper method to process target after UI update"""
        self.dog.target = text
        self.dog.ai_client.set_target(text)
        self.target_set = True
        
        # Start search
        self.start_search()
        
        # Play TTS after everything else
        if self.dog.env["tts"]:
            QTimer.singleShot(300, lambda: self.play_tts(response))

    def execute_feedback_action(self, assistant):
        """Helper method to execute feedback actions"""
        # First play TTS
        if self.dog.env["tts"]:
            self.dog.ai_client.tts(assistant.action)
        
        # Then execute action
        self.dog.activate_sportclient(
            assistant.action, 
            int(assistant.move), 
            int(assistant.shift), 
            int(assistant.turn)
        )
        
        # Show completion message and resume search after action
        QTimer.singleShot(2000, self.complete_feedback)

    def complete_feedback(self):
        """Helper method to complete feedback process"""
        resume_msg = "Feedback processed. Returning to search mode..."
        self.add_robot_message(resume_msg)
        
        # Play TTS after UI update
        if self.dog.env["tts"]:
            QTimer.singleShot(300, lambda: self.play_tts(resume_msg))
        
        # Resume search after message and TTS
        QTimer.singleShot(600, lambda: self.resume_auto_mode())

    def resume_auto_mode(self):
        """Helper method to resume auto mode"""
        self.feedback_mode = False
        self.dog.feedback_complete_event.set()

    def add_user_message(self, text):
        message = ChatMessage(text, is_user=True)
        self.chat_layout.addWidget(message)
        self._scroll_to_bottom()

    def add_robot_message(self, text, image=None):
        message = ChatMessage(text, is_user=False, image=image)
        self.chat_layout.addWidget(message)
        self._scroll_to_bottom()

    def _scroll_to_bottom(self):
        QTimer.singleShot(100, lambda: self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        ))

    def start_search(self):
        """Helper method to start search process"""
        # Start camera thread
        self.camera_thread = CameraThread(self.dog)
        self.camera_thread.frame_update.connect(self.update_camera_feed)
        self.camera_thread.start()

        # Start search thread
        self.search_thread = SearchThread(self.dog)
        self.search_thread.status_update.connect(self.handle_status_update)
        self.search_thread.start()
        self.search_started = True

    def handle_status_update(self, status, image=None):
        self.add_robot_message(status, image)

    def update_camera_feed(self, image):
        if self.search_started:
            self.latest_frame = image

    def closeEvent(self, event):
        """Handle cleanup when closing the window"""
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
        if hasattr(self, 'dog'):
            self.dog.shutdown()
        event.accept()

class CameraThread(QThread):
    frame_update = pyqtSignal(QImage)

    def __init__(self, dog_instance):
        super().__init__()
        self.dog = dog_instance
        self.running = True

    def run(self):
        while self.running:
            frame = self.dog.read_frame()
            if frame is not None:
                # Convert PIL image to QImage
                frame_array = np.array(frame)
                height, width, channel = frame_array.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_array.data, width, height, bytes_per_line, 
                               QImage.Format.Format_RGB888)
                self.frame_update.emit(q_image)
            self.msleep(30)

    def stop(self):
        self.running = False

class SearchThread(QThread):
    status_update = pyqtSignal(str, QImage)

    def __init__(self, dog_instance):
        super().__init__()
        self.dog = dog_instance

    def run(self):
        # Modify the robot dog's response handling to emit updates
        original_get_response = self.dog.ai_client.get_response_by_LLM

        def get_response_wrapper(*args, **kwargs):
            response = original_get_response(*args, **kwargs)
            if response:
                # Convert the latest frame to QImage
                frame = args[0]  # Assuming the first argument is the frame
                frame_array = np.array(frame)
                height, width, channel = frame_array.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_array.data, width, height, bytes_per_line, 
                               QImage.Format.Format_RGB888)
                
                # Emit both the response and the frame
                self.status_update.emit(
                    f"Action: {response.action}\nReason: {response.reason}", 
                    q_image
                )
            return response

        # Replace the original method with our wrapper
        self.dog.ai_client.get_response_by_LLM = get_response_wrapper
        
        # Run the search
        self.dog.run_gpt() 