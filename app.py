from flask import Flask
from flask_cors import CORS

from api import register_routes

def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/gemini-response": {"origins": "http://localhost:5173"}})
    register_routes(app)
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)