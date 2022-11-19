import hashlib
from dotenv import dotenv_values

config = dotenv_values(".env")


def makeHash(email):
    email = email + config["KEYWORD"]
    h = hashlib.sha256(email.encode())
    return h.hexdigest()
