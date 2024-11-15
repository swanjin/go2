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
        self.pending_feedback_action = None
        self.awaiting_feedback = False

        self.processing_timer = QTimer()
        self.processing_timer.timeout.connect(self.update_processing_animation)
        self.processing_dots = 0

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

        # Feedback confirmation buttons
        self.confirm_widget = QWidget()
        self.confirm_widget.setStyleSheet("""
            QWidget {
                background-color: #F1F3F4;
                border-top: 1px solid #E0E0E0;
            }
        """)
        confirm_layout = QHBoxLayout(self.confirm_widget)
        confirm_layout.setContentsMargins(20, 20, 20, 20)
        
        self.yes_button = QPushButton("Yes")
        self.no_button = QPushButton("No")
        self.yes_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 20px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        self.no_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border-radius: 20px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #da190b;
            }
        """)
        
        self.yes_button.clicked.connect(self.confirm_feedback)
        self.no_button.clicked.connect(self.reject_feedback)
        
        confirm_layout.addWidget(self.yes_button)
        confirm_layout.addWidget(self.no_button)
        self.confirm_widget.hide()
        layout.addWidget(self.confirm_widget)

        # Add status buttons container
        self.status_buttons = QWidget()
        self.status_buttons.setStyleSheet("""
            QWidget {
                background-color: white;
                border-top: 1px solid #E0E0E0;
            }
        """)
        status_layout = QHBoxLayout(self.status_buttons)
        status_layout.setContentsMargins(20, 10, 20, 10)
        
        # Create status buttons
        status_questions = [
            ("🤖 Robot Status", "Robot Status"),  # (button text, message to send)
            ("💭 Feedback", "feedback")
        ]
        
        for button_text, message in status_questions:
            btn = QPushButton(button_text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #E3F2FD;
                    color: #1A73E8;
                    border: 2px solid #1A73E8;
                    border-radius: 20px;
                    padding: 10px 20px;
                    font-size: 14px;
                    font-weight: bold;
                    min-width: 150px;
                    margin: 0 10px;
                }
                QPushButton:hover {
                    background-color: #BBDEFB;
                    color: #0D47A1;
                    border: 2px solid #0D47A1;
                }
                QPushButton:pressed {
                    background-color: #90CAF9;
                    padding: 11px 19px 9px 21px;
                }
            """)
            btn.clicked.connect(lambda checked, m=message: self.send_status_question(m))
            status_layout.addWidget(btn)

        self.status_buttons.hide()
        layout.addWidget(self.status_buttons)

        # Input area
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
        
        self.input_widget.hide()
        layout.addWidget(self.input_widget)

    def start_conversation(self):
        self.start_button.hide()
        self.start_button.deleteLater()
        self.input_widget.show()
        
        welcome_message = "Hello! I'm Go2, your robot dog assistant. What would you like me to find for you?"
        self.add_robot_message(welcome_message)
        
        self.conversation_started = True
        self.message_input.setFocus()
        
        if self.dog.env["tts"]:
            QTimer.singleShot(100, lambda: self.dog.ai_client.tts(welcome_message))

    def update_processing_animation(self):
        self.processing_dots = (self.processing_dots + 1) % 4
        dots = "." * self.processing_dots
        self.processing_label.setText(f"Processing{dots.ljust(3)}")

    def start_processing_animation(self):
        self.processing_label = QLabel("Processing...")
        self.processing_label.setStyleSheet("""
            QLabel {
                color: #1A73E8;
                font-size: 14px;
                font-weight: bold;
                padding: 10px;
            }
        """)
        self.chat_layout.addWidget(self.processing_label)
        self.processing_timer.start(500)  # Update every 500ms
        self._scroll_to_bottom()

    def stop_processing_animation(self):
        self.processing_timer.stop()
        if hasattr(self, 'processing_label'):
            self.processing_label.deleteLater()
            self.processing_label = None

    def send_message(self):
        if not self.conversation_started:
            return
            
        text = self.message_input.text().strip()
        if not text:
            return
        
        #self.add_user_message(text)
        #self.message_input.clear()

        status_questions = [
            "what are you doing",
            "what's happening",
            "current status",
            "status",
            "what is happening",
            "what's going on",
            "how is it going",
            "found anything",
            "see anything",
            "how are you doing"
        ]

        if any(q in text.lower() for q in status_questions):
            self.add_user_message(text)
            self.message_input.clear()
            if self.target_set:
                status = f"I'm currently searching for {self.dog.target}. "
                if self.feedback_mode:
                    status += "I'm in feedback mode, waiting for your guidance."
                else:
                    status += "I'm in automatic search mode."
            else:
                status = "I'm waiting for you to tell me what to search for."
            
            self.add_robot_message(status)
            
        elif not self.target_set:
            self.add_user_message(text)
            self.message_input.clear()
            if "apple" in text.lower():
                response = f"I'll start searching for apple now."
                self.add_robot_message(response)
                self.status_buttons.show()
                QTimer.singleShot(100, lambda: self.process_target("apple", response))
                QTimer.singleShot(1000, self.start_processing_animation)
            else:
                clarify_msg = "Please tell me specifically about the target you want me to find."
                self.add_robot_message(clarify_msg)
                if self.dog.env["tts"]:
                    QTimer.singleShot(300, lambda: self.play_tts(clarify_msg))
                
        elif text.lower() == "feedback":
            self.add_user_message(text)
            self.message_input.clear()
            self.dog.feedback_complete_event.clear()
            self.dog.interrupt_round_flag.set()
            self.feedback_mode = True
            self.awaiting_feedback = True
            
            feedback_msg = "Feedback mode activated. Please provide your feedback."
            self.add_robot_message(feedback_msg)
            
            if self.dog.env["tts"]:
                QTimer.singleShot(300, lambda: self.play_tts(feedback_msg))
            
        elif self.feedback_mode and self.awaiting_feedback:

            self.add_user_message(text)
            self.message_input.clear()
            QApplication.processEvents()
            self.status_buttons.hide()

            assistant = self.dog.ai_client.get_response_by_feedback(text)


            if assistant:
                self.pending_feedback_action = assistant
                
                action_descriptions = []
                for action in assistant.action:
                    if 'move' in action:
                        times = assistant.move
                        action_descriptions.append(f"<b>{action}</b> <b>{times}</b> time{'s' if times > 1 else ''}")
                    elif 'shift' in action:
                        times = assistant.shift
                        action_descriptions.append(f"<b>{action}</b> <b>{times}</b> time{'s' if times > 1 else ''}")
                    elif 'turn' in action:
                        times = assistant.turn
                        action_descriptions.append(f"<b>{action}</b> <b>{times}</b> time{'s' if times > 1 else ''}")

                confirmation_msg = "I understand you want me to " + " and ".join(action_descriptions) + ". Is this correct?"
                self.add_robot_message(confirmation_msg)

                self.confirm_widget.show()
                self.input_widget.hide()
                self.awaiting_feedback = False
                
                if self.dog.env["tts"]:
                    QTimer.singleShot(300, lambda: self.play_tts(confirmation_msg))
        else:
            self.add_user_message(text)
            self.message_input.clear()
            self.dog.feedback = text

    def confirm_feedback(self):
        """사용자가 해석된 피드백을 승인할 때"""
        if self.pending_feedback_action:
            print("Pending feedback action:", self.pending_feedback_action)
            response_text = f"Executing your request..."
            self.status_buttons.show()
            self.add_robot_message(response_text)
            
            # 피드백 액션을 직접 변수에 저장
            action_to_execute = self.pending_feedback_action
            
            # UI 상태 초기화
            self.confirm_widget.hide()
            self.input_widget.show()
            
            # 액션 실행
            self.execute_feedback_action(action_to_execute)
            
            # 상태 초기화
            self.pending_feedback_action = None
            self.awaiting_feedback = False

    def reject_feedback(self):
        """사용자가 해석된 피드백을 거부할 때"""
        reject_msg = "Please provide more specific feedback about what you want me to do."
        self.add_robot_message(reject_msg)
        
        if self.dog.env["tts"]:
            QTimer.singleShot(300, lambda: self.play_tts(reject_msg))
        
        self.confirm_widget.hide()
        self.input_widget.show()
        self.pending_feedback_action = None
        self.awaiting_feedback = True

    def play_tts(self, message):
        if self.dog.env["tts"]:
            self.dog.ai_client.tts(message)

    def process_target(self, text, response):
        self.dog.target = text
        self.dog.ai_client.set_target(text)
        self.target_set = True
        
        self.start_search()
        
        if self.dog.env["tts"]:
            QTimer.singleShot(300, lambda: self.play_tts(response))

    def execute_feedback_action(self, assistant):
        """피드백 액션을 실행하는 메소드"""
        try:
            if not assistant:
                print("Error: No assistant object provided")
                return
            
            print("Executing feedback with assistant:", assistant)  # 디버그 출력
            
            # action 확인
            if not hasattr(assistant, 'action'):
                print("Error: Assistant has no action attribute")
                return
            
            # TTS 실행 (action이 있는 경우에만)
            if self.dog.env["tts"] and assistant.action:
                action_text = assistant.action[0] if isinstance(assistant.action, list) else assistant.action
                self.dog.ai_client.tts(action_text)
            
            # 로봇 동작 실행
            self.dog.activate_sportclient(
                assistant.action,
                int(assistant.move) if hasattr(assistant, 'move') else 0,
                int(assistant.shift) if hasattr(assistant, 'shift') else 0,
                int(assistant.turn) if hasattr(assistant, 'turn') else 0
            )
            
            # 실행 완료 처리
            QTimer.singleShot(2000, self.complete_feedback)
            
        except Exception as e:
            print(f"Error in execute_feedback_action: {str(e)}")
            error_msg = f"Failed to execute feedback: {str(e)}"
            self.add_robot_message(error_msg)
            self.resume_auto_mode()

    def complete_feedback(self):
        resume_msg = "Feedback processed. Returning to search mode..."
        self.add_robot_message(resume_msg)
        
        if self.dog.env["tts"]:
            QTimer.singleShot(300, lambda: self.play_tts(resume_msg))
        
        QTimer.singleShot(600, lambda: self.resume_auto_mode())

    def resume_auto_mode(self):
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
        self.camera_thread = CameraThread(self.dog)
        self.camera_thread.frame_update.connect(self.update_camera_feed)
        self.camera_thread.start()

        self.search_thread = SearchThread(self.dog)
        self.search_thread.status_update.connect(self.handle_status_update)
        self.search_thread.start()
        self.search_started = True

    def handle_status_update(self, status, image=None):
        self.stop_processing_animation()
        self.add_robot_message(status, image)

    def update_camera_feed(self, image):
        if self.search_started:
            self.latest_frame = image

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
        if hasattr(self, 'dog'):
            self.dog.shutdown()
        event.accept()

    def send_status_question(self, question):
        """Handle status button clicks by sending the question as a message"""
        if not self.conversation_started:
            return
        
        self.message_input.setText(question)
        self.send_message()

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
        original_get_response = self.dog.ai_client.get_response_by_LLM

        def get_response_wrapper(*args, **kwargs):
            response = original_get_response(*args, **kwargs)
            if response:
                frame = args[0]
                frame_array = np.array(frame)
                height, width, channel = frame_array.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_array.data, width, height, bytes_per_line, 
                               QImage.Format.Format_RGB888)
                
                self.status_update.emit(
                    f"Action: {response.action}\nReason: {response.reason}", 
                    q_image
                )
            return response

        self.dog.ai_client.get_response_by_LLM = get_response_wrapper
        self.dog.run_gpt() 