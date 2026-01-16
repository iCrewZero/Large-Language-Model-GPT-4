class ToolRouter:
    def route(self, text):
        if "calculate" in text:
            return "calculator"
        return None
