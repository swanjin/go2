from queue import Queue
import threading

class TTSManager:
    def __init__(self):
        self.current_tts = None
        self.is_enabled = False
        self.lock = threading.Lock()
    
    def enable(self):
        """TTS 기능 활성화"""
        self.is_enabled = True
    
    def disable(self):
        """TTS 기능 비활성화"""
        self.is_enabled = False
        self.stop_current()
    
    def play(self, text, ai_client):
        """새로운 TTS 재생"""
        if not self.is_enabled:
            return
            
        # 현재 재생 중인 TTS가 있다면 중단
        self.stop_current()
        
        with self.lock:
            try:
                # 새로운 TTS 시작
                self.current_tts = ai_client.tts(text)
            except Exception as e:
                print(f"TTS Error: {e}")
    
    def stop_current(self):
        """현재 재생 중인 TTS 중단"""
        with self.lock:
            if self.current_tts:
                try:
                    self.current_tts.stop()  # TTS 중단 메서드 (AI 클라이언트에 구현 필요)
                except Exception as e:
                    print(f"Error stopping TTS: {e}")
                finally:
                    self.current_tts = None
