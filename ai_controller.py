import openai_client

class AiController:
    @staticmethod
    def getClient(env: str, key):
        if env["ai"] == "openai":
            return openai_client.OpenaiClient(env, key)
