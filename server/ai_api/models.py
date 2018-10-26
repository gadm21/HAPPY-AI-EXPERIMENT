from sqlalchemy import DateTime
from sqlalchemy.sql import func

from ai_api import db


class ANPR(db.Model):
    __tablename__ = 'anpr'

    id = db.Column(db.Integer, primary_key=True)
    plate_number = db.Column(db.String())
    url = db.Column(db.String())
    timestamp = db.Column(DateTime(timezone=True), server_default=func.now())

    def __init__(self, number, url):
        self.url = url
        self.number = number
