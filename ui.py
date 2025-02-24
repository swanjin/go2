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

class TTSWorker(QThread):
    def __init__(self, text, dog, parent=None):
        super().__init__(parent)
        self.text = text
        self.dog = dog  # dog 인스턴스는 openai_client (TTS 메서드가 포함된)를 가지고 있음

    def run(self):
        # TTS 시작 로그
        print(f"[TTSWorker] Starting TTS with text: {self.text}")
        
        # 실제 TTS 실행 (openai_client.py의 tts() 메서드)
        self.dog.ai_client.tts(self.text)
        
        # TTS 종료 로그
        print("[TTSWorker] TTS finished")

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
            # 기존 메시지 표시 코드는 그대로 유지
            message = QLabel(text)
            message.setWordWrap(True)
            message.setStyleSheet(Styles.chat_message_bubble(is_user))
            container_layout.addWidget(message)
            
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
    confirm_feedback_signal = pyqtSignal()
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
        
        # 컨펌 상태일 때의 처리
        if self.message_data.confirming_action and self.message_data.pending_feedback_action:
            if self.dog.ai_client.is_yes(text):
                print("✅ Yes 응답 받음 - 액션 실행 시작")
                self.show_loading_signal.emit()  # 로딩 표시
                self.execute_feedback_action_signal.emit(self.message_data.pending_feedback_action)
                self.message_data.pending_feedback_action = None
                self.message_data.confirming_action = False
            else:
                print("❌ No 응답 받음")
                if self.dog.ai_client.is_no_command(text):
                    print("❌ 단순 거절 - 상세 설명 요청")
                    self.show_loading_signal.emit()  # 로딩 표시
                    self.add_robot_message_signal.emit("Please provide more details or clarify your feedback.", None)
                    self.message_data.confirming_action = False
                else:
                    print("🔄 새로운 명령 감지 - 액션 생성 중")
                    self.show_loading_signal.emit()  # 로딩 표시
                    frame = self.dog.read_frame()
                    image_bboxes_array, image_detected_objects, image_distances, image_description = self.dog.ai_client.feedback_mode_on(frame)
                    assistant = self.dog.ai_client.get_response_landmark_or_general_command(
                        text, 
                        image_bboxes_array, 
                        image_detected_objects, 
                        image_distances, 
                        image_description
                    )
                    self.message_data.pending_feedback_action = assistant.action
                    self.confirm_feedback_signal.emit()
            return
        
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
                if self.dog.env["interactive"]:
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
                self.message_data.pending_feedback_action = assistant.action
                self.confirm_feedback_signal.emit()
            
            else:
                print("Getting answer to question from AI client...")  # Debug print
                answer = self.dog.ai_client.get_response_non_command(text)
                print(f"AI answer received: {answer}")  # Debug print
                self.add_robot_message_signal.emit(answer, None)

        else:
            print("Type 'feedback' to give feedback")  # Debug print
            self.dog.feedback = text

class RobotDogUI(QMainWindow):
    # 클래스 레벨에서 시그널 정의
    confirm_feedback_signal = pyqtSignal()
    
    def __init__(self, dog_instance):
        super().__init__()
        self.dog = dog_instance
        self.message_data = MessageData()
        self.search_started = False
        self.loading_message = None
        self.loading_timer = None
        self.delayed_loading_timer = None  # 지연된 로딩을 위한 타이머 추가
        
        # 시그널 연결
        self.confirm_feedback_signal.connect(self.confirm_feedback)
        
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
        
        # 피드백 확인 중인 경우 특별 처리
        if self.message_data.confirming_action and self.message_data.pending_feedback_action:
            # 스레드를 통해 메시지 처리
            self.send_message_thread = SendMessageThread(self.message_data, self.dog)
            self.send_message_thread.add_user_message_signal.connect(self.add_user_message)
            self.send_message_thread.add_robot_message_signal.connect(self.add_robot_message_with_loading)
            self.send_message_thread.confirm_feedback_signal.connect(self.confirm_feedback)
            self.send_message_thread.execute_feedback_action_signal.connect(self.execute_feedback_action)
            self.send_message_thread.show_loading_signal.connect(self.show_loading)
            self.send_message_thread.start()
            
            self.message_input.clear()
            return
        
        # 기존 피드백 모드에서의 처리
        elif self.message_data.feedback_mode and self.message_data.awaiting_feedback:
            # 먼저 사용자 메시지를 표시
            self.add_user_message(self.message_data.text)
            
            if self.message_data.text.lower() != "feedback mode":
                self.show_loading()
                print("[DEBUG] show_loading - send_message (feedback mode)")
            
            self.send_message_thread = SendMessageThread(self.message_data, self.dog)
            self.send_message_thread.process_target_signal.connect(self.process_target)
            self.send_message_thread.input_widget_signal.connect(self.input_widget.setVisible)
            self.send_message_thread.feedback_button_signal.connect(self.feedback_button.setVisible)
            self.send_message_thread.confirm_feedback_signal.connect(self.confirm_feedback)
            self.send_message_thread.add_robot_message_signal.connect(self.add_robot_message_with_loading)
            self.send_message_thread.activate_feedback_mode_signal.connect(self.activate_feedback_mode)
            self.send_message_thread.execute_feedback_action_signal.connect(self.execute_feedback_action)
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
        self.send_message_thread.confirm_feedback_signal.connect(self.confirm_feedback)
        self.send_message_thread.add_robot_message_signal.connect(self.add_robot_message_with_loading)
        self.send_message_thread.activate_feedback_mode_signal.connect(self.activate_feedback_mode)
        self.send_message_thread.execute_feedback_action_signal.connect(self.execute_feedback_action)
        self.send_message_thread.start()

    def confirm_feedback(self):
        if self.message_data.pending_feedback_action:
            print("Pending feedback action found.")
            action_to_execute = self.message_data.pending_feedback_action

            # 액션 설명 포맷팅
            if not action_to_execute:
                action_description = ""
            else:
                formatted_actions = []
                for action, group in groupby(action_to_execute):
                    count = len(list(group))
                    formatted_actions.append(f"{action} {count} times")
                action_description = " and ".join(formatted_actions)

            # 로딩 애니메이션 숨기기
            self.hide_loading()
            print("[DEBUG] hide_loading - confirm_feedback")
            print(f"피드백 액션: {action_to_execute}")

            is_landmark_action = self.dog.ai_client.is_landmark_action
            
            try:
                # GPT에게 랜드마크 판단을 요청
                landmark_name = None
                if is_landmark_action:
                    landmark_name = self.dog.ai_client.get_landmark_name(self.message_data.text)

                print(f"Is landmark action_ui: {is_landmark_action}")
                print(f"Found landmark name: {landmark_name}")  # 디버깅용

                # 로봇 메시지 표시
                if is_landmark_action and landmark_name:
                    self.add_robot_message(
                        f"Yes, I can go to the {landmark_name}. Should I try?"
                    )
                else:
                    self.add_robot_message(
                        f"I understand you want me to {action_description}. "
                        "Should I proceed with this action?"
                    )

                # 피드백 대기 상태 유지
                self.message_data.awaiting_feedback = True
                self.message_data.confirming_action = True
                self.message_input.setFocus()

            except Exception as e:
                print(f"Error in confirm_feedback: {e}")
                # 에러 발생시 기본 메시지 표시
                self.add_robot_message(
                    f"I understand you want me to {action_description}. "
                    "Should I proceed with this action?"
                )
                self.message_data.awaiting_feedback = True
                self.message_data.confirming_action = True
                self.message_input.setFocus()

        else:
            print("No pending feedback action.")

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
            
            print("Executing feedback with actions:", action)
            self.show_loading()
            print("[DEBUG] show_loading - execute_feedback_action")
            self.dog.activate_sportclient(action)
            QTimer.singleShot(3000, self.complete_feedback)
            
        except Exception as e:
            print(f"Error in execute_feedback_action: {str(e)}")
            error_msg = Messages.ERROR_FEEDBACK_EXECUTION.format(str(e))
            self.add_robot_message(error_msg)
            self.resume_auto_mode()

    def complete_feedback(self):
        # 먼저 로딩 애니메이션 숨기기
        self.hide_loading()
        print("[DEBUG] hide_loading - complete_feedback")
        
        # 그 다음 피드백 완료 메시지 표시
        self.add_robot_message(Messages.FEEDBACK_COMPLETE)
        
        QTimer.singleShot(600, lambda: self.resume_auto_mode())

    def resume_auto_mode(self):
        self.feedback_mode = False
        self.message_data.feedback_mode = False
        self.dog.feedback_complete_event.set()
        
        QTimer.singleShot(0, self._scroll_to_bottom)

        if self.dog.env["interactive"]:
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
        message = ChatMessage(text, is_user=False, image=image)
        self.chat_layout.addWidget(message)
        QTimer.singleShot(0, self._scroll_to_bottom)
        if text == Messages.SEARCH_COMPLETE.format(self.dog.target):
            return
        else:
            if not self.message_data.feedback_mode and text != Messages.WELCOME:
                self.tts_thread = TTSWorker(text, self.dog)
                self.tts_thread.finished.connect(self.on_tts_finished)
                self.tts_thread.start()

    def on_tts_finished(self):
        self.dog.tts_finished_event.set()
        if not self.message_data.feedback_mode:
            # Ensure the delayed_loading_timer is initialized
            if self.delayed_loading_timer is None:
                self.delayed_loading_timer = QTimer(self)  # Initialize the timer if it's None
            
            # Stop any existing timer
            self.delayed_loading_timer.stop()
            
            # Set up the delayed loading timer
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

        self.search_thread = SearchThread(self.dog, self)
        self.search_thread.status_update.connect(self.handle_status_update)
        self.search_thread.end_search.connect(self.handle_end_search)
        self.search_thread.start()
        self.search_started = True

    def handle_status_update(self, status, image=None):
        print("handle_status_update called")
        print(f"Feedback mode: {self.message_data.feedback_mode}")
        if self.dog.env["woz"] or self.dog.env["vn"]:
            return
        else:
            self.hide_loading()
            self.add_robot_message(status, image)

    def handle_end_search(self, message, delayed_time=81000):
        if self.dog.env["woz"]:
            print("[DEBUG] hide_loading - handle_end_search")
            QTimer.singleShot(delayed_time - 100, lambda: self.hide_loading())
            QTimer.singleShot(delayed_time, lambda: self.add_robot_message(
                Messages.SEARCH_COMPLETE.format(self.dog.target)
            ))
            # TTS도 delayed_time + 100ms 후에 실행
            QTimer.singleShot(delayed_time + 100, lambda: self._delayed_tts(
                Messages.SEARCH_COMPLETE.format(self.dog.target)
            ))
        else:
            self.hide_loading()
            self.add_robot_message(Messages.SEARCH_COMPLETE.format(self.dog.target))

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
        self.send_message_thread.confirm_feedback_signal.connect(self.confirm_feedback)
        self.send_message_thread.add_robot_message_signal.connect(self.add_robot_message_with_loading)
        self.send_message_thread.activate_feedback_mode_signal.connect(self.activate_feedback_mode)
        self.send_message_thread.execute_feedback_action_signal.connect(self.execute_feedback_action)
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
        """로딩 메시지를 표시합니다."""
        if self.loading_message:
            self.loading_message.deleteLater()
        self.loading_message = ChatMessage(is_user=False, is_loading=True)
        self.chat_layout.addWidget(self.loading_message)
        self._scroll_to_bottom()
    
    def hide_loading(self):
        """로딩 메시지를 제거합니다."""
        if self.loading_message:
            self.loading_message.deleteLater()
            self.loading_message = None

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

    def __init__(self, dog_instance, ui_instance):
        super().__init__()
        self.dog = dog_instance
        self.ui = ui_instance  # Add a reference to the UI instance

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
                combined_message = f"I'm going to {formatted_action}. {response.reason}"

                if self.dog.ai_client.memory_list:
                    last_round = self.dog.ai_client.memory_list[-1]
                else:
                    print("No rounds in memory_list")

                if self.dog.env["interactive"]:
                    if last_round.assistant.initial_state == (0,0,0):
                        print("🤖 Robot is asking for help.")  
                        last_round.round_number = last_round.round_number - 1
                        detected_objects = ", ".join(last_round.detected_objects)
                        combined_message = Messages.FEEDBACK_HELP.format(detected_objects)
                        self.dog.tts_finished_event.clear()
                        self.status_update.emit(combined_message, q_image)
                        self.dog.tts_finished_event.wait()
                        # Start feedback mode automatically
                        self.ui.trigger_feedback_mode()  # Use the UI instance to call the method
                    else:
                        self.dog.tts_finished_event.clear()
                        self.status_update.emit(combined_message, q_image)
                        self.dog.tts_finished_event.wait()
                
                if  self.dog.env["woz"] or self.dog.env["vn"]:
                    self.status_update.emit(combined_message, q_image)

                if self.dog.env["woz"] or formatted_action == 'stop':
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