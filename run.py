from app import create_app
from config import Config

app = create_app()

if __name__ == "__main__":
    # use_reloader=False keeps the camera thread from being started twice in debug mode
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG, use_reloader=False)
