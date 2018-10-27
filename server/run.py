import os

from ai_api import app

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000, debug=os.environ.get('FLASK_DEBUG', False))
