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

        # # Feedback confirmation buttons
        # self.confirm_widget = QWidget()
        # self.confirm_widget.setStyleSheet("""
        #     QWidget {
        #         background-color: #F1F3F4;
        #         border-top: 1px solid #E0E0E0;
        #     }
        # """)
        # confirm_layout = QHBoxLayout(self.confirm_widget)
        # confirm_layout.setContentsMargins(20, 20, 20, 20)
        
        # self.yes_button = QPushButton("Yes")
        # self.no_button = QPushButton("No")
        # self.yes_button.setStyleSheet("""
        #     QPushButton {
        #         background-color: #4CAF50;
        #         color: white;
        #         border-radius: 20px;
        #         padding: 10px 20px;
        #         font-weight: bold;
        #     }
        #     QPushButton:hover {
        #         background-color: #45a049;
        #     }
        # """)
        # self.no_button.setStyleSheet("""
        #     QPushButton {
        #         background-color: #f44336;
        #         color: white;
        #         border-radius: 20px;
        #         padding: 10px 20px;
        #         font-weight: bold;
        #     }
        #     QPushButton:hover {
        #         background-color: #da190b;
        #     }
        # """)
        
        # self.yes_button.clicked.connect(self.confirm_feedback)
        # self.no_button.clicked.connect(self.reject_feedback)
        
        # confirm_layout.addWidget(self.yes_button)
        # confirm_layout.addWidget(self.no_button)
        # self.confirm_widget.hide()
        # layout.addWidget(self.confirm_widget)

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

        # Feedback button
        self.feedback_button = QPushButton("💭 Feedback")
        self.feedback_button.setStyleSheet("""
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
        self.feedback_button.clicked.connect(self.trigger_feedback_mode)
        self.feedback_button.hide()
        layout.addWidget(self.feedback_button, alignment=Qt.AlignmentFlag.AlignCenter)

    def start_conversation(self):
        self.start_button.hide()
        self.start_button.deleteLater()
        self.input_widget.show()
        
        welcome_message = "Hello! I'm Go2, your robot dog assistant. What would you like me to find for you?"
        self.add_robot_message(welcome_message)
        
        self.conversation_started = True
        self.message_input.setFocus()
        
        if self.dog.env["tts"]:
            QTimer.singleShot(100, lambda: self.play_tts(welcome_message))

    def send_message(self):
        print("\n=== send_message called ===")  # Debug print
        if not self.conversation_started:
            print("Conversation not started")  # Debug print
            return
        
        text = self.message_input.text().strip()
        if not text:
            print("Empty message")  # Debug print
            return
        
        # Display the user message immediately
        self.add_user_message(text)
        
        print(f"Processing message: '{text}'")  # Debug print
        print(f"Current mode - Feedback mode: {self.feedback_mode}, Awaiting feedback: {self.awaiting_feedback}")  # Debug print
        
        # Proceed with processing logic
        if not self.target_set:
            print("Processing target setting")  # Debug print

            if "apple" in text.lower():
                response = f"I'll start searching for apple now."
                self.add_robot_message(response)
                QTimer.singleShot(100, lambda: self.process_target("apple", response))
                QTimer.singleShot(1000, self.start_processing_animation)
                # self.feedback_button.show()

            else:
                clarify_msg = "Apologies, I didn't catch that. Could you please clarify the target you'd like me to identify?"
                self.add_robot_message(clarify_msg)
                if self.dog.env["tts"]:
                    QTimer.singleShot(300, lambda: self.play_tts(clarify_msg))
                
        elif text.lower() == "feedback":
            print("Activating feedback mode")  # Debug print
            self.stop_processing_animation()
            self.dog.feedback_complete_event.clear()
            self.dog.interrupt_round_flag.set()
            self.feedback_mode = True
            self.awaiting_feedback = True
            
            feedback_msg = "Gotcha! Feedback mode activated. Please provide your feedback."
            self.add_robot_message(feedback_msg)
            
            # Hide feedback button when feedback mode is activated
            self.feedback_button.hide()
                       
            if self.dog.env["tts"]:
                QTimer.singleShot(300, lambda: self.play_tts(feedback_msg))
            
            self.dog.ai_client.history_log_file.write(f"\n=== Conversation ===\n")
            self.dog.ai_client.history_log_file.flush()

        elif self.feedback_mode and self.awaiting_feedback:
            print("\n=== Processing feedback in UI ===")  # Debug print
            print(f"Feedback text: '{text}'")  # Debug print
            self.dog.ai_client.history_log_file.write(f"User: {text} \n")
            self.dog.ai_client.history_log_file.flush()

            frame = self.dog.read_frame()
            print(f"Frame received: {frame is not None}")  # Debug print
            image_array_bboxes, image_description = self.dog.ai_client.feedback_mode_on(frame)

            if self.dog.ai_client.is_feedback_mode_exit(text):
                print("Exit command received")  # Debug print
                exit_msg = "Alright, I'll wrap up feedback mode and switch back to automatic search."
                self.add_robot_message(exit_msg)
                
                if self.dog.env["tts"]:
                    QTimer.singleShot(300, lambda: self.play_tts(exit_msg))
                
                self.awaiting_feedback = False
                self.resume_auto_mode()
            
            elif text.endswith("!"):
                print("❗ Processing feedback with exclamation mark")              
                confirmation_msg, assistant = self.dog.ai_client.feedback_to_action(text, image_array_bboxes, image_description)
                print(f"🤖 AI interpreted action: {assistant.action if hasattr(assistant, 'action') else 'None'}")
                self.pending_feedback_action = assistant
                self.add_robot_message(confirmation_msg)
                if self.dog.env["tts"]:
                    QTimer.singleShot(300, lambda: self.play_tts(confirmation_msg))
                
                self.confirm_feedback()
                # self.confirm_widget.show()
                # self.input_widget.hide()
                # print("✅ Waiting for user confirmation...")
                self.awaiting_feedback = False
            
            else:
                print("Getting answer to question from AI client...")  # Debug print
                answer = self.dog.ai_client.get_response_by_feedback(text)
                print(f"AI answer received: {answer}")  # Debug print
                self.add_robot_message(answer)
                if self.dog.env["tts"]:
                    QTimer.singleShot(300, lambda: self.play_tts(answer))
                self.dog.ai_client.history_log_file.write(f"Go2: {answer} \n")
                self.dog.ai_client.history_log_file.flush()

        else:
            print("Type 'feedback' to give feedback")  # Debug print
            self.dog.feedback = text

    def update_processing_animation(self):
        self.processing_dots = (self.processing_dots + 1) % 4
        dots = "." * self.processing_dots
        self.processing_label.setText(f"Processing{dots.ljust(3)}")

    def start_processing_animation(self):
        if not hasattr(self, 'processing_label') or self.processing_label is None:
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
        if hasattr(self, 'processing_label') and self.processing_label is not None:
            self.processing_label.deleteLater()
            self.processing_label = None
    
    def confirm_feedback(self):
        """사용자가 해석된 피드백을 승인할 때"""
        if self.pending_feedback_action:
            print("Pending feedback action:", self.pending_feedback_action)  # 디버그 출력
            # response_text = "I'm going to process your feedback."
            # self.add_robot_message(response_text)
            # if self.dog.env["tts"]:
            #     QTimer.singleShot(300, lambda: self.play_tts(response_text))

            # 피드백 액션을 직접 변수에 저장
            action_to_execute = self.pending_feedback_action
            
            # # UI 상태 초기화
            # self.confirm_widget.hide()
            # self.input_widget.show()
            
            # 액션 실행
            self.execute_feedback_action(action_to_execute)
            
            # 상태 초기화
            self.pending_feedback_action = None
            self.awaiting_feedback = False

    def reject_feedback(self):
        """사용자가 해석된 피드백을 거부할 때"""
        reject_msg = """It seems my suggested actions don't align with your needs. Could you clarify your expectations or suggest adjustments?"""
        self.add_robot_message(reject_msg)
        self.dog.ai_client.openai_params_for_text["messages"] = self.dog.ai_client.openai_params_for_text["messages"][:-1]
        self.dog.ai_client.openai_params_for_text["messages"].append({"role": "assistant", "content": reject_msg})
        self.dog.ai_client.history_log_file.write(f"Go2: {reject_msg} \n")
        self.dog.ai_client.history_log_file.flush()

        if self.dog.env["tts"]:
            QTimer.singleShot(300, lambda: self.play_tts(reject_msg))
        
        self.confirm_widget.hide()
        self.input_widget.show()
        self.pending_feedback_action = None
        self.awaiting_feedback = True

    def play_tts(self, message):
        if self.dog.env["tts"]:
            try:
                print(f"Playing TTS for message: {message}")  # Debug print
                self.dog.ai_client.tts(message)
            except Exception as e:
                print(f"TTS Error: {e}")

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
            
            # 로봇 동작 실행
            self.dog.activate_sportclient(
                assistant.action,
                int(assistant.move),
                int(assistant.shift),
                int(assistant.turn)
            )
            
            # 실행 완료 처리
            QTimer.singleShot(3000, self.complete_feedback)
            
        except Exception as e:
            print(f"Error in execute_feedback_action: {str(e)}")
            error_msg = f"Failed to execute feedback: {str(e)}"
            self.add_robot_message(error_msg)
            self.resume_auto_mode()

    def complete_feedback(self):
        resume_msg = "I'm back to automatic search mode now."
        self.add_robot_message(resume_msg)
        
        if self.dog.env["tts"]:
            QTimer.singleShot(300, lambda: self.play_tts(resume_msg))
        
        QTimer.singleShot(600, lambda: self.resume_auto_mode())

    def resume_auto_mode(self):
        self.feedback_mode = False
        self.dog.ai_client.reset_messages_feedback()
        self.dog.feedback_complete_event.set()

        QTimer.singleShot(1000, self.start_processing_animation)
        # self.feedback_button.show()
        QTimer.singleShot(0, self._scroll_to_bottom)

    def add_user_message(self, text):
        message = ChatMessage(text, is_user=True)
        self.chat_layout.addWidget(message)
        QTimer.singleShot(0, self._scroll_to_bottom)
        self.message_input.clear()

    def add_robot_message(self, text, image=None):
        message = ChatMessage(text, is_user=False, image=image)
        self.chat_layout.addWidget(message)
        QTimer.singleShot(0, self._scroll_to_bottom)

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
        # self.search_thread.tts_request.connect(self.play_tts)
        # self.search_thread.start_processing.connect(self.start_processing_animation)
        # self.search_thread.stop_processing.connect(self.stop_processing_animation)
        self.search_thread.start()
        self.search_started = True

    def handle_status_update(self, status, image=None):
        self.add_robot_message(status, image)
        self.stop_processing_animation()

    def update_camera_feed(self, image):
        if self.search_started:
            self.latest_frame = image

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread'):
            self.camera_thread.stop()
        if hasattr(self, 'dog'):
            self.dog.shutdown()
        event.accept()

    def trigger_feedback_mode(self):
        """Simulate typing 'feedback' and trigger send_message."""
        self.message_input.setText("feedback")
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
    # start_processing = pyqtSignal()  # Signal to start processing animation
    # stop_processing = pyqtSignal()   # Signal to stop processing animation

    def __init__(self, dog_instance):
        super().__init__()
        self.dog = dog_instance

    def format_actions(self, actions):
        if isinstance(actions, list):
            return ' and then '.join(map(str, actions))
        return str(actions)

    def run(self):
        original_get_response = self.dog.ai_client.get_response_by_LLM

        def get_response_wrapper(*args, **kwargs):
            # self.start_processing.emit()  # Emit signal to start processing animation

            response = original_get_response(*args, **kwargs)
            if response:
                frame = args[0]
                frame_array = np.array(frame)
                height, width, channel = frame_array.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_array.data, width, height, bytes_per_line, 
                               QImage.Format.Format_RGB888)
                
                formatted_action = self.format_actions(response.action)
                combined_message = f"I'm going to {formatted_action}. {response.reason}"

                self.status_update.emit(combined_message, q_image)

            # self.stop_processing.emit()  # Emit signal to stop processing animation
            return response

        self.dog.ai_client.get_response_by_LLM = get_response_wrapper
        self.dog.run_gpt() 