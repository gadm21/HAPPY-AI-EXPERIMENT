import os

import connexion
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from ai_api.config import config


def create_app():
    app = connexion.App(__name__)
    app.app.config.from_object(config[os.environ.get("APP_MODE", "development")])
    app.add_api("swagger.yaml")
    return app.app


app = create_app()
db = SQLAlchemy(app)
migrate = Migrate(app, db)

from ai_api.models import ANPR
