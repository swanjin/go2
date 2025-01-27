class Colors:
    PRIMARY = "#1A73E8"
    PRIMARY_DARK = "#1557AA"
    PRIMARY_LIGHT = "#E3F2FD"
    SECONDARY = "#F1F3F4"
    BACKGROUND = "white"
    TEXT_PRIMARY = "#333333"
    TEXT_SECONDARY = "#666666"
    TEXT_PLACEHOLDER = "#999999"
    BORDER = "#E0E0E0"
    SUCCESS = "#4CAF50"
    ERROR = "#f44336"
    GRAY_LIGHT = "#F8F9FA"  # 로봇 메시지용 밝은 회색
    GRAY = "#E9ECEF"        # 더 진한 회색이 필요한 경우

class Sizes:
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 800
    FONT_LARGE = 32
    FONT_MEDIUM = 28
    FONT_SMALL = 26
    PADDING_LARGE = 20
    PADDING_MEDIUM = 15
    PADDING_SMALL = 10
    BORDER_RADIUS = 20
    MESSAGE_MAX_WIDTH = 600
    CIRCLE_SIZE = 20
    CIRCLE_INNER_MARGIN = 5
    CHAT_MESSAGE_MAX_WIDTH = 600
    IMAGE_MAX_WIDTH = 400
    IMAGE_MAX_HEIGHT = 300
    CHAT_MARGIN = (10, 5, 10, 5)  # left, top, right, bottom
    CONTAINER_MARGIN = (0, 0, 0, 0)

class Styles:
    @staticmethod
    def main_window():
        return f"""
            QMainWindow {{
                background-color: {Colors.BACKGROUND};
            }}
        """

    @staticmethod
    def primary_button():
        return f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border-radius: {Sizes.BORDER_RADIUS}px;
                padding: {Sizes.PADDING_MEDIUM}px {Sizes.PADDING_LARGE}px;
                font-size: {Sizes.FONT_LARGE}px;
                font-weight: bold;
                margin: {Sizes.PADDING_MEDIUM}px;
            }}
            QPushButton:hover {{
                background-color: {Colors.PRIMARY_DARK};
            }}
        """

    @staticmethod
    def feedback_button():
        return f"""
            QPushButton {{
                background-color: {Colors.PRIMARY_LIGHT};
                color: {Colors.PRIMARY};
                border: 2px solid {Colors.PRIMARY};
                border-radius: {Sizes.BORDER_RADIUS}px;
                padding: 12px 18px;
                font-size: 25px;
                font-weight: bold;
                min-width: 150px;
                margin: 10px;
            }}
            QPushButton:hover {{
                background-color: {Colors.PRIMARY_LIGHT};
                color: {Colors.PRIMARY_DARK};
                border: 2px solid {Colors.PRIMARY_DARK};
            }}
        """

    @staticmethod
    def input_field():
        return f"""
            QLineEdit {{
                border: 1px solid {Colors.BORDER};
                border-radius: {Sizes.BORDER_RADIUS}px;
                padding: {Sizes.PADDING_SMALL}px {Sizes.PADDING_MEDIUM}px;
                background-color: white;
                color: {Colors.TEXT_PRIMARY};
                font-size: {Sizes.FONT_MEDIUM}px;
                min-height: 20px;
            }}
            QLineEdit:focus {{
                border: 1px solid {Colors.PRIMARY};
                outline: none;
            }}
            QLineEdit::placeholder {{
                color: {Colors.TEXT_PLACEHOLDER};
            }}
        """

    @staticmethod
    def chat_message_sender(is_user=False):
        return f"""
            QLabel {{
                color: {'#1A73E8' if is_user else '#666666'};
                font-weight: bold;
                font-size: {Sizes.FONT_SMALL}px;
            }}
        """

    @staticmethod
    def chat_message_bubble(is_user=False):
        return f"""
            QLabel {{
                background-color: {'#1A73E8' if is_user else Colors.GRAY};
                color: {'white' if is_user else Colors.TEXT_PRIMARY};
                border-radius: 15px;
                padding: {Sizes.PADDING_SMALL}px {Sizes.PADDING_MEDIUM}px;
                font-size: {Sizes.FONT_MEDIUM}px;
            }}
        """

    @staticmethod
    def chat_image():
        return """
            QLabel {
                border-radius: 10px;
                margin-top: 5px;
            }
        """

    @staticmethod
    def scroll_area():
        return f"""
            QScrollArea {{
                border: none;
                background-color: {Colors.BACKGROUND};
            }}
        """

    @staticmethod
    def input_widget():
        return f"""
            QWidget {{
                background-color: {Colors.SECONDARY};
                border-top: 1px solid {Colors.BORDER};
            }}
        """

    @staticmethod
    def send_button():
        return f"""
            QPushButton {{
                background-color: {Colors.PRIMARY};
                color: white;
                border-radius: {Sizes.BORDER_RADIUS}px;
                padding: {Sizes.PADDING_SMALL}px {Sizes.PADDING_MEDIUM}px;
                font-size: {Sizes.FONT_MEDIUM}px;
                font-weight: bold;
                border: none;
            }}
            QPushButton:hover {{
                background-color: {Colors.PRIMARY_DARK};
            }}
        """

    @staticmethod
    def processing_label():
        return f"""
            QLabel {{
                color: black;
                font-size: {Sizes.FONT_LARGE}px;
                font-weight: bold;
                padding: {Sizes.PADDING_SMALL}px;
            }}
        """
