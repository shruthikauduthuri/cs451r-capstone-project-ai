from flask import Flask, jsonify, request

from llm_chatbot import generate_gemini_response


def register_routes(app: Flask) -> None:
    @app.route("/")
    def home():
        return jsonify({"message": "Server is running"})

    @app.route("/gemini-response", methods=["POST"])
    def gemini_response():
        data = request.get_json(silent=True)

        if not data or "message" not in data:
            return jsonify({"error": "Invalid request. 'message' is required."}), 400

        user_message = data["message"].strip()
        if not user_message:
            return jsonify({"error": "Message cannot be empty."}), 400

        try:
            response_text = generate_gemini_response(user_message)
            return jsonify({"success": True, "response": response_text}), 200
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
