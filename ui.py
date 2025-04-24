from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QLabel, QTextEdit, QLineEdit,
                           QScrollArea, QFrame, QMessageBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QPainter, QPen
from PIL import Image
import numpy as np
import math
from itertools import groupby

import sys
import cv2
from dataclasses import dataclass
from ui_config import Colors, Sizes, Styles
from messages import Messages
from navi_config import NaviConfig
import os

# Set CUDA environment variables before any CUDA operations
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# If you want to use a specific GPU, set it here (e.g., "0" for the first GPU)
# If you want to disable CUDA, set it to "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Also add this function to initialize CUDA properly
def initialize_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            # Force initialization of CUDA in the main thread
            _ = torch.zeros(1).cuda()
            print(f"CUDA initialized successfully. Available devices: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available. Using CPU.")
    except Exception as e:
        print(f"Error initializing CUDA: {str(e)}")
        print("Falling back to CPU.")

class TTSWorker(QThread):
    def __init__(self, text, dog, parent=None):
        super().__init__(parent)
        self.text = text
        self.dog = dog  # dog 인스턴스는 openai_client (TTS 메서드가 포함된)를 가지고 있음

    def run(self):
        # UI 업데이트와 TTS 사이에 약간의 지연 추가
        import time
        time.sleep(0.2)  # 0.2초 지연 (0.5초에서 줄임)
        
        # TTS 시작 로그
        print(f"[TTSWorker] Starting TTS with text: {self.text}")
        
        try:
            # 실제 TTS 실행 (openai_client.py의 tts() 메서드)
            self.dog.ai_client.tts(self.text)
            # TTS 종료 로그
            print("[TTSWorker] TTS finished")
        except Exception as e:
            print(f"[TTSWorker] Error during TTS: {str(e)}")

class ChatMessage(QFrame):
    def __init__(self, text="", is_user=False, image=None, is_loading=False, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        
        # Main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(*Sizes.CHAT_MARGIN)
        
        # Message container
        container = QFrame()
        container.setMaximumWidth(Sizes.CHAT_MESSAGE_MAX_WIDTH)
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(*Sizes.CONTAINER_MARGIN)
        
        # Sender label
        sender = QLabel("You" if is_user else "Go2")
        sender.setStyleSheet(Styles.chat_message_sender(is_user))
        container_layout.addWidget(sender)
        
        if is_loading:
            # 로딩 인디케이터 추가
            loading = SimpleLoadingIndicator()
            container_layout.addWidget(loading)
        else:
            # 이미지가 있으면 먼저 표시
            if image is not None:
                img_label = QLabel()
                pixmap = QPixmap.fromImage(image)
                scaled_pixmap = pixmap.scaled(
                    Sizes.IMAGE_MAX_WIDTH, 
                    Sizes.IMAGE_MAX_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                img_label.setPixmap(scaled_pixmap)
                img_label.setStyleSheet(Styles.chat_image())
                container_layout.addWidget(img_label)
            
            # 텍스트 메시지 표시
            message = QLabel(text)
            message.setWordWrap(True)
            message.setStyleSheet(Styles.chat_message_bubble(is_user))
            container_layout.addWidget(message)

        # Align messages
        main_layout.addStretch() if is_user else None
        main_layout.addWidget(container)
        main_layout.addStretch() if not is_user else None
        
@dataclass
class MessageData:
    text: str = ""
    feedback_mode: bool = False
    awaiting_feedback: bool = False
    target_set: bool = False
    conversation_started: bool = False
    pending_feedback_action: any = None
    confirming_action: bool = False

class SendMessageThread(QThread):
    process_target_signal = pyqtSignal(str, str)
    input_widget_signal = pyqtSignal(bool)
    feedback_button_signal = pyqtSignal(bool)
    add_user_message_signal = pyqtSignal(str)
    add_robot_message_signal = pyqtSignal(str, object)
    activate_feedback_mode_signal = pyqtSignal()
    execute_feedback_action_signal = pyqtSignal(list)
    show_loading_signal = pyqtSignal()  # 로딩 시그널 추가

    def __init__(self, message_data: MessageData, dog_instance, parent=None):
        super().__init__(parent)
        self.message_data = message_data
        self.dog = dog_instance

    def run(self):
        if not self.message_data.conversation_started:
            return
        
        text = self.message_data.text.strip()
        if not text:
            return
        
        # 먼저 사용자 메시지를 표시
        self.add_user_message_signal.emit(text)
        
        print("\n=== send_message called ===")  # Debug print
        if not self.message_data.conversation_started:
            print("Conversation not started")  # Debug print
            return
        
        text = self.message_data.text.strip()
        if not text:
            print("Empty message")  # Debug print
            return
        
        # Display the user message immediately
        self.add_user_message_signal.emit(text)
        
        print(f"Processing message: '{text}'")  # Debug print
        print(f"Current mode - Feedback mode: {self.message_data.feedback_mode}, Awaiting feedback: {self.message_data.awaiting_feedback}")  # Debug print
        
        # Proceed with processing logic
        if not self.message_data.target_set:
            print("Processing target setting")  # Debug print

            if "apple" in text.lower():
                response = Messages.SEARCH_START.format("apple")
                self.add_robot_message_signal.emit(response, None)
                self.process_target_signal.emit("apple", response)
                self.input_widget_signal.emit(False)
                if self.dog.env["i"]:
                    self.feedback_button_signal.emit(True)

            else:
                clarify_msg = Messages.ERROR_NO_TARGET
                self.add_robot_message_signal.emit(clarify_msg, None)
                    
        elif text.lower() == "feedback mode":
            print("Activating feedback mode")  # Debug print
            self.dog.feedback_complete_event.clear()
            self.dog.interrupt_round_flag.set()
            self.feedback_button_signal.emit(False)
            self.input_widget_signal.emit(True)
            QTimer.singleShot(100, self.activate_feedback_mode_signal.emit)
            
        elif self.message_data.feedback_mode and self.message_data.awaiting_feedback:
            print("\n=== Processing feedback in UI ===")  # Debug print
            print(f"Feedback text: '{text}'")  # Debug print
            frame = self.dog.read_frame()
            print(f"Frame received: {frame is not None}")  # Debug print
            image_bboxes_array, image_detected_objects, image_distances, image_description = self.dog.ai_client.feedback_mode_on(frame)

            if self.dog.ai_client.is_instruction_command(text):
                print("❗ Executing instruction or command")            
                assistant = self.dog.ai_client.get_response_landmark_or_general_command(text, image_bboxes_array, image_detected_objects, image_distances, image_description)
                print(f"📋 생성된 액션: {assistant.action}")
                # 확인 과정 없이 바로 액션 실행
                self.show_loading_signal.emit()
                self.execute_feedback_action_signal.emit(assistant.action)
            
            else:
                print("Getting answer to question from AI client...")  # Debug print
                answer = self.dog.ai_client.get_response_non_command(text)
                print(f"AI answer received: {answer}")  # Debug print
                self.add_robot_message_signal.emit(answer, None)

        else:
            print("Type 'feedback' to give feedback")  # Debug print
            self.dog.feedback = text

class RobotDogUI(QMainWindow):
    def __init__(self, dog_instance):
        super().__init__()
        self.dog = dog_instance
        self.message_data = MessageData()
        self.search_started = False
        self.loading_message = None
        self.loading_timer = None
        self.delayed_loading_timer = None
        
        # Initialize CUDA before any operations that might use it
        initialize_cuda()
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Go2 Chat Interface')
        self.setGeometry(100, 100, Sizes.WINDOW_WIDTH, Sizes.WINDOW_HEIGHT)
        self.setStyleSheet(Styles.main_window())

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
        self.start_button.setStyleSheet(Styles.primary_button())
        self.start_button.clicked.connect(self.start_conversation)
        self.chat_layout.addWidget(self.start_button, alignment=Qt.AlignmentFlag.AlignCenter)

        # Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.chat_area)
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet(Styles.scroll_area())
        layout.addWidget(self.scroll)

        # Input area
        self.input_widget = QWidget()
        self.input_widget.setStyleSheet(Styles.input_widget())
        input_layout = QHBoxLayout(self.input_widget)
        input_layout.setContentsMargins(20, 20, 20, 20)
        
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Send a message...")
        self.message_input.setStyleSheet(Styles.input_field())
        self.message_input.returnPressed.connect(self.send_message)
        
        send_button = QPushButton("Send")
        send_button.setStyleSheet(Styles.send_button())
        send_button.clicked.connect(self.send_message)
        
        input_layout.addWidget(self.message_input)
        input_layout.addWidget(send_button)
        
        self.input_widget.hide()
        layout.addWidget(self.input_widget)

        # Feedback button
        self.feedback_button = QPushButton("💬 Feedback Mode")
        self.feedback_button.setStyleSheet(Styles.feedback_button())
        self.feedback_button.clicked.connect(self.trigger_feedback_mode)
        self.feedback_button.hide()
        layout.addWidget(self.feedback_button, alignment=Qt.AlignmentFlag.AlignCenter)

    def start_conversation(self):
        self.start_button.deleteLater()
        self.input_widget.hide()

        self.add_robot_message(Messages.WELCOME)
        self.message_data.conversation_started = True
        
        # TTS가 끝나면 input_widget을 보여주도록 설정
        self.tts_thread = TTSWorker(Messages.WELCOME, self.dog)
        self.tts_thread.finished.connect(self.show_input_after_welcome)
        self.tts_thread.start()

    def show_input_after_welcome(self):
        self.input_widget.show()
        self.message_input.setFocus()
        self.dog.tts_finished_event.set()

    def send_message(self):
        self.message_data.text = self.message_input.text()
        
        # 피드백 모드에서의 처리
        if self.message_data.feedback_mode and self.message_data.awaiting_feedback:
            # 먼저 사용자 메시지를 표시
            self.add_user_message(self.message_data.text)
            
            if self.message_data.text.lower() != "feedback mode":
                self.show_loading()
                print("[DEBUG] show_loading - send_message (feedback mode)")
            
            self.send_message_thread = SendMessageThread(self.message_data, self.dog)
            # 시그널 연결
            self.send_message_thread.process_target_signal.connect(self.process_target)
            self.send_message_thread.input_widget_signal.connect(self.input_widget.setVisible)
            self.send_message_thread.feedback_button_signal.connect(self.feedback_button.setVisible)
            self.send_message_thread.add_robot_message_signal.connect(self.add_robot_message_with_loading)
            self.send_message_thread.activate_feedback_mode_signal.connect(self.activate_feedback_mode)
            self.send_message_thread.execute_feedback_action_signal.connect(self.execute_feedback_action)
            self.send_message_thread.show_loading_signal.connect(self.show_loading)
            self.send_message_thread.start()
            
            self.message_input.clear()
            return
        
        # 기존 send_message 로직 계속 진행
        # 1. 먼저 사용자 메시지를 표시
        self.add_user_message(self.message_data.text)
        
        # 2. 피드백 모드로 처음 진입하는 경우가 아닐 때만 로딩 표시
        if self.message_data.text.lower() != "feedback mode":
            self.show_loading()
            print("[DEBUG] show_loading - send_message (normal message)")
        
        self.send_message_thread = SendMessageThread(self.message_data, self.dog)
        self.send_message_thread.process_target_signal.connect(self.process_target)
        self.send_message_thread.input_widget_signal.connect(self.input_widget.setVisible)
        self.send_message_thread.feedback_button_signal.connect(self.feedback_button.setVisible)
        self.send_message_thread.add_robot_message_signal.connect(self.add_robot_message_with_loading)
        self.send_message_thread.activate_feedback_mode_signal.connect(self.activate_feedback_mode)
        self.send_message_thread.execute_feedback_action_signal.connect(self.execute_feedback_action)
        self.send_message_thread.show_loading_signal.connect(self.show_loading)
        self.send_message_thread.start()

    def process_target(self, text, response):
        print()
        self.dog.target = text
        self.dog.ai_client.set_target(text)
        self.message_data.target_set = True
        
        # 지연된 로딩 타이머 설정
        self.delayed_loading_timer = QTimer(self)
        self.delayed_loading_timer.timeout.connect(lambda: self.show_loading())
        self.delayed_loading_timer.setSingleShot(True)
        self.delayed_loading_timer.start(1000)
        print("[DEBUG] show_loading - process_target (delayed timer set)")
        
        QTimer.singleShot(1000, self.start_search)

    def execute_feedback_action(self, action):
        try:
            if not action:
                print("Error: No actions provided")
                return
            
            # 액션 설명 포맷팅 - 특별한 패턴 확인
            formatted_actions = []
            
            # 액션 그룹화 및 특별 패턴 확인
            for action_item, group in groupby(action):
                count = len(list(group))
                
                # 특별한 패턴 처리: turn right 또는 turn left가 두 번 연속이면 turn around로 변경
                if (action_item == "turn right" or action_item == "turn left") and count == 2:
                    formatted_actions.append("turn around")
                else:
                    if count > 1:
                        formatted_actions.append(f"{action_item} {count} times")
                    else:
                        formatted_actions.append(f"{action_item}")
                    
            action_description = " and ".join(formatted_actions)
            
            print(f"피드백 액션: {action}")
            
            # 랜드마크 관련 액션인지 확인
            is_landmark_action = self.dog.ai_client.is_landmark_action
            self.hide_loading()
            
            # 피드백 메시지 표시 - 더 자연스러운 표현으로 개선
            try:
                # 랜드마크 관련 액션인 경우
                if is_landmark_action:
                    landmark_names = self.dog.ai_client.get_multiple_landmark_names(self.message_data.text)
                    print(f"Is landmark action_ui: {is_landmark_action}")
                    print(f"Found landmark names: {landmark_names}")
                    
                    if landmark_names:
                        if len(landmark_names) > 1:
                            landmarks_text = " and ".join(landmark_names)
                            self.add_robot_message(f"Got it! I'll head to the area between {landmarks_text} now.")
                        else:
                            self.add_robot_message(f"Thanks for the feedback! I'll go to the {landmark_names[0]} right away.")
                    else:
                        # 랜드마크 관련이지만 이름이 없는 경우 일반 메시지 사용
                        import random
                        general_messages = [
                            "Thanks for the help! I’ll do it just like you said.",
                            "Okay, I’ll change it the way you told me to!",
                            "Got it! I’ll follow your directions now.",
                            "I’ll work on that right away—thanks for the tip!",
                            "That helps a lot! I’m on it!"
                        ]
                        self.add_robot_message(random.choice(general_messages))
                else:
                    # 일반 명령에 대한 다양한 응답
                    import random
                    general_messages = [
                        "I'll follow your instructions right away.",
                        "Thanks for guiding me! I'm on it.",
                        "Got it! I'm working on it right now.",
                        "Your feedback is clear. I'm doing it now."
                    ]
                    self.add_robot_message(random.choice(general_messages))
            except Exception as e:
                print(f"Error generating feedback message: {e}")
                self.add_robot_message("I've received your feedback and I'm acting on it now.")
            
            # UI 업데이트를 즉시 처리하도록 강제
            QApplication.processEvents()
            
            # 약간의 지연 후 액션 실행 (UI가 확실히 업데이트된 후)
            QTimer.singleShot(300, lambda: self._execute_action(action))
            
        except Exception as e:
            print(f"Error in execute_feedback_action: {str(e)}")
            error_msg = Messages.ERROR_FEEDBACK_EXECUTION.format(str(e))
            self.add_robot_message(error_msg)
            self.resume_auto_mode()

    def _execute_action(self, action):
        """액션을 실행하는 별도의 메서드"""
        print("Executing feedback with actions:", action)
        self.dog.activate_sportclient(action)
        QTimer.singleShot(3000, self.complete_feedback)

    def complete_feedback(self):
        # 먼저 로딩 애니메이션 숨기기
        QTimer.singleShot(600, lambda: self.resume_auto_mode())

    def resume_auto_mode(self):
        self.feedback_mode = False
        self.message_data.feedback_mode = False
        self.dog.feedback_complete_event.set()
        
        QTimer.singleShot(0, self._scroll_to_bottom)

        if self.dog.env["i"]:
            self.input_widget.hide()
            self.feedback_button.show()

        self.show_loading()
        print("[DEBUG] show_loading - resume_auto_mode")

    def add_user_message(self, text):
        message = ChatMessage(text, is_user=True)
        self.chat_layout.addWidget(message)
        QTimer.singleShot(0, self._scroll_to_bottom)
        self.message_input.clear()

    def add_robot_message(self, text, image=None):
        # 먼저 UI에 메시지 추가
        message = ChatMessage(text, is_user=False, image=image)
        self.chat_layout.addWidget(message)
        
        # UI 업데이트를 즉시 처리하도록 강제
        QApplication.processEvents()
        
        # 스크롤을 아래로 이동
        self._scroll_to_bottom()
        
        # 특정 메시지는 TTS 처리하지 않음
        if text == Messages.SEARCH_COMPLETE.format(self.dog.target):
            return
        
        # 피드백 모드가 아니고 환영 메시지가 아닌 경우에만 TTS 실행
        if not self.message_data.feedback_mode and text != Messages.WELCOME:
            try:
                # TTS 스레드 시작
                self.tts_thread = TTSWorker(text, self.dog)
                self.tts_thread.finished.connect(self.on_tts_finished)
                self.tts_thread.start()
            except Exception as e:
                print(f"[TTS] Error starting TTS: {str(e)}")
                # TTS에 실패해도 UI 흐름은 계속되도록 함
                self.on_tts_finished()

    def on_tts_finished(self):
        self.dog.tts_finished_event.set()
        if not self.message_data.feedback_mode:
            # 지연된 로딩 타이머 설정
            if self.delayed_loading_timer is not None:
                self.delayed_loading_timer.stop()
            self.delayed_loading_timer = QTimer(self)
            self.delayed_loading_timer.timeout.connect(lambda: self.show_loading())
            self.delayed_loading_timer.setSingleShot(True)
            self.delayed_loading_timer.start(1000)
            print("[DEBUG] show_loading - on_tts_finished (delayed timer set)")

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
        self.search_thread.end_search.connect(self.handle_end_search)
        self.search_thread.start()
        self.search_started = True

    def handle_status_update(self, status, image=None):
        print("handle_status_update called")
        print(f"Feedback mode: {self.message_data.feedback_mode}")
        if self.dog.env["w"] or self.dog.env["v"]:
            return
        else:
            self.hide_loading()
            self.add_robot_message(status, image)

    def handle_end_search(self, message, delayed_time=61000):
        if self.dog.env["w"]:
            print("[DEBUG] hide_loading - handle_end_search")
            QTimer.singleShot(delayed_time - 100, lambda: self.hide_loading())
            QTimer.singleShot(delayed_time, lambda: self.add_robot_message(
                Messages.SEARCH_COMPLETE.format(self.dog.target)
            ))
            QTimer.singleShot(delayed_time + 1000, lambda: self._delayed_tts(
                Messages.SEARCH_COMPLETE.format(self.dog.target)
            ))
            # 검색 완료 후 모든 기능 종료
            QTimer.singleShot(delayed_time + 2000, self.shutdown_all_features)
        else:
            self.hide_loading()
            self.add_robot_message(Messages.SEARCH_COMPLETE.format(self.dog.target))
            # 검색 완료 후 모든 기능 종료
            QTimer.singleShot(500, self.shutdown_all_features)

    def _delayed_tts(self, text):
        """TTS를 실행하는 헬퍼 메서드"""
        self.tts_thread = TTSWorker(text, self.dog)
        self.tts_thread.start()

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
        # 지연된 로딩 타이머가 있다면 취소
        if self.delayed_loading_timer is not None:
            self.delayed_loading_timer.stop()
            self.delayed_loading_timer = None
            print("[DEBUG] Cancelled delayed loading timer")
        
        # 기존 로딩 애니메이션 제거
        self.hide_loading()
        print("[DEBUG] hide_loading - trigger_feedback_mode")
        
        # 예약된 로딩 타이머가 있다면 취소
        if self.loading_timer is not None:
            self.loading_timer.stop()
            self.loading_timer = None
        
        self.message_input.setText("feedback mode")
        # 피드백 모드 진입 시에는 로딩 애니메이션 없이 메시지만 전송
        self.message_data.text = self.message_input.text()
        self.add_user_message(self.message_data.text)
        
        self.send_message_thread = SendMessageThread(self.message_data, self.dog)
        self.send_message_thread.process_target_signal.connect(self.process_target)
        self.send_message_thread.input_widget_signal.connect(self.input_widget.setVisible)
        self.send_message_thread.feedback_button_signal.connect(self.feedback_button.setVisible)
        self.send_message_thread.add_robot_message_signal.connect(self.add_robot_message_with_loading)
        self.send_message_thread.activate_feedback_mode_signal.connect(self.activate_feedback_mode)
        self.send_message_thread.execute_feedback_action_signal.connect(self.execute_feedback_action)
        self.send_message_thread.show_loading_signal.connect(self.show_loading)
        self.send_message_thread.start()
        
        QTimer.singleShot(100, self.activate_feedback_mode)
        self.message_input.setFocus()

    def activate_feedback_mode(self):
        """Activate feedback mode after a delay."""
        self.message_data.feedback_mode = True
        self.message_data.awaiting_feedback = True

    def trigger_exit_mode(self):
        """Simulate typing 'exit feedback mode' and trigger send_message."""
        self.message_input.setText("exit feedback mode")
        self.send_message()

    def trigger_execute_mode(self):
        """Simulate typing 'execute feedback' and trigger send_message."""
        self.message_input.setText("execute feedback")
        self.send_message()

    def show_feedback_mode_message(self):
        if self.feedback_mode:
            feedback_mode_label = QLabel(Messages.FEEDBACK_MODE_ACTIVATED)
            feedback_mode_label.setStyleSheet(f"""
                QLabel {{
                    color: {Colors.PRIMARY};
                    font-size: {Sizes.FONT_MEDIUM}px;
                    font-weight: bold;
                    padding: {Sizes.PADDING_SMALL}px;
                }}
            """)
            self.chat_layout.addWidget(feedback_mode_label)
            self._scroll_to_bottom()

    def hide_feedback_mode_message(self):
        """Hide feedback mode activated message."""
        if self.feedback_mode:
            # Logic to remove the feedback message from the layout
            for i in reversed(range(self.chat_layout.count())):
                widget = self.chat_layout.itemAt(i).widget()
                if isinstance(widget, QLabel) and "Feedback mode activated" in widget.text():
                    widget.deleteLater()

    def show_auto_mode_message(self):
        if self.search_started:
            auto_mode_label = QLabel(Messages.AUTO_MODE_ACTIVATED)
            auto_mode_label.setStyleSheet(f"""
                QLabel {{
                    color: {Colors.PRIMARY};
                    font-size: {Sizes.FONT_MEDIUM}px;
                    font-weight: bold;
                    padding: {Sizes.PADDING_SMALL}px;
                }}
            """)
            self.chat_layout.addWidget(auto_mode_label)
            self._scroll_to_bottom()

    def hide_auto_mode_message(self):
        """Hide auto mode activated message."""
        if self.search_started:
            # Logic to remove the auto mode message from the layout
            # Assuming you have a way to identify and remove the specific QLabel
            for i in reversed(range(self.chat_layout.count())):
                widget = self.chat_layout.itemAt(i).widget()
                if isinstance(widget, QLabel) and "Auto search mode activated" in widget.text():
                    widget.deleteLater()

    def show_loading(self):
        # 이미 로딩 메시지가 있으면 새로 만들지 않음
        if self.loading_message:
            return
        
        # 로딩 메시지 생성 및 표시
        self.loading_message = ChatMessage(is_loading=True, is_user=False)
        self.chat_layout.addWidget(self.loading_message)
        
        # 로딩 시작 시 피드백 버튼도 함께 표시 (i 모드인 경우에만)
        if self.dog.env["i"] and not self.message_data.feedback_mode:
            self.feedback_button.show()
        
        QApplication.processEvents()  # UI 업데이트 즉시 처리
        QTimer.singleShot(0, self._scroll_to_bottom)

    def hide_loading(self):
        # 로딩 메시지가 있으면 제거
        if self.loading_message:
            self.loading_message.deleteLater()
            self.loading_message = None
            
            # 로딩이 끝날 때 피드백 버튼도 함께 숨김
            if self.dog.env["i"] and not self.message_data.feedback_mode:
                self.feedback_button.hide()
            
            QApplication.processEvents()  # UI 업데이트 즉시 처리
            QTimer.singleShot(0, self._scroll_to_bottom)

    def add_robot_message_with_loading(self, text, image=None):
        """로딩을 숨기고 로봇 메시지를 표시합니다."""
        self.hide_loading()
        print("[DEBUG] hide_loading - add_robot_message_with_loading")
        self.add_robot_message(text, image)

    def show_confirmation_dialog(self, action_description):
        """Show a confirmation dialog for the action."""
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Question)
        msg_box.setWindowTitle("Confirm Action")
        msg_box.setText(f"Do you want to execute the following action?\n\n{action_description}")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        return msg_box.exec()

    def shutdown_all_features(self):
        """모든 UI 기능을 종료하는 메서드"""
        print("[DEBUG] Shutting down all UI features")
        
        # 입력 위젯 비활성화
        self.input_widget.setEnabled(False)
        self.input_widget.hide()
        
        # 피드백 버튼 비활성화
        self.feedback_button.setEnabled(False)
        self.feedback_button.hide()
        
        # 진행 중인 모든 스레드 종료
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread.wait()
        
        if hasattr(self, 'search_thread') and self.search_thread.isRunning():
            self.search_thread.quit()
            self.search_thread.wait()
        
        if hasattr(self, 'send_message_thread') and self.send_message_thread.isRunning():
            self.send_message_thread.quit()
            self.send_message_thread.wait()
        
        if hasattr(self, 'tts_thread') and self.tts_thread.isRunning():
            self.tts_thread.quit()
            self.tts_thread.wait()
        
        # 타이머 종료
        if self.loading_timer is not None:
            self.loading_timer.stop()
        
        if self.delayed_loading_timer is not None:
            self.delayed_loading_timer.stop()
        
        # 로딩 메시지 제거
        self.hide_loading()
        
        # 검색 완료 메시지 표시
        completion_message = QLabel("Search completed. All features are now disabled.")
        completion_message.setStyleSheet(f"""
            QLabel {{
                color: {Colors.PRIMARY};
                font-size: {Sizes.FONT_MEDIUM}px;
                font-weight: bold;
                padding: {Sizes.PADDING_SMALL}px;
                margin-top: 20px;
            }}
        """)
        self.chat_layout.addWidget(completion_message, alignment=Qt.AlignmentFlag.AlignCenter)
        self._scroll_to_bottom()
        
        # 메시지 데이터 초기화
        self.message_data.feedback_mode = False
        self.message_data.awaiting_feedback = False
        self.message_data.confirming_action = False
        
        print("[DEBUG] All UI features have been shut down")

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
    end_search = pyqtSignal(str)

    def __init__(self, dog_instance):
        super().__init__()
        self.dog = dog_instance

    def format_actions(self,actions):
        if not actions:  # Handle empty list case
            return ""

        formatted_actions = []
        
        for action, group in groupby(actions):
            count = len(list(group))
            if count > 1:
                formatted_actions.append(f"{action} {count} times")
            else:
                formatted_actions.append(action)

        return " and ".join(formatted_actions)

    def run(self):
        original_get_response = self.dog.ai_client.get_response_by_LLM

        def get_response_wrapper(*args, **kwargs):
            # 라운드가 시작됨을 표시
            print("get_response_wrapper called")
            # LLM에게 응답을 요청
            response = original_get_response(*args, **kwargs)

            if response:
                frame = args[0]
                frame_array = np.array(frame)
                height, width, channel = frame_array.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_array.data, width, height, bytes_per_line, 
                                QImage.Format.Format_RGB888)
                
                formatted_action = self.format_actions(response.action)
                combined_message = f"{response.reason}"

                if self.dog.env["i"]:
                    self.dog.tts_finished_event.clear()
                    self.status_update.emit(combined_message, q_image)
                    self.dog.tts_finished_event.wait()
                
                if  self.dog.env["w"] or self.dog.env["v"]:
                    self.status_update.emit(combined_message, q_image)

                if self.dog.env["w"] or formatted_action == 'stop':
                    end_message = Messages.SEARCH_COMPLETE.format("apple")
                    self.end_search.emit(end_message)

            return response

        self.dog.ai_client.get_response_by_LLM = get_response_wrapper
        self.dog.run_gpt()


class SimpleLoadingIndicator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(80, 40)
        
        # 기본 레이아웃 설정
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # 3개의 점 생성
        self.dots = []
        for _ in range(3):
            dot = QLabel("•")
            dot.setStyleSheet(f"""
                QLabel {{
                    color: {Colors.PRIMARY};
                    font-size: 24px;
                }}
            """)
            layout.addWidget(dot)
            self.dots.append(dot)
        
        # 애니메이션 타이머
        self.current_dot = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_dots)
        self.timer.start(300)
        
    def animate_dots(self):
        # 모든 점을 흐리게
        for dot in self.dots:
            dot.setStyleSheet(f"""
                QLabel {{
                    color: {Colors.TEXT_PLACEHOLDER};
                    font-size: 24px;
                }}
            """)
        
        # 현재 점만 강조
        self.dots[self.current_dot].setStyleSheet(f"""
            QLabel {{
                color: {Colors.PRIMARY};
                font-size: 24px;
            }}
        """)
        
        # 다음 점으로 이동
        self.current_dot = (self.current_dot + 1) % 3 