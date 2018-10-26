import os

from ai_api import app

if __name__ == '__main__':
    app.run(port=8080, debug=os.environ.get('FLASK_DEBUG', False))
