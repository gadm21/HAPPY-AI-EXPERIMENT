import os

config = {
    "development": "ai_api.config.DevelopmentConfig",
    "testing": "ai_api.config.TestingConfig",
    "production": "ai_api.config.ProductionConfig",
    "staging": "ai_api.config.StagingConfig",
}


class Config(object):
    ENV = "default"
    TESTING = False
    SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]
    SQLALCHEMY_TRACK_MODIFICATIONS = False


class ProductionConfig(Config):
    ENV = "production"


class StagingConfig(Config):
    ENV = "staging"


class DevelopmentConfig(Config):
    ENV = "development"
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ["DATABASE_URL"]
