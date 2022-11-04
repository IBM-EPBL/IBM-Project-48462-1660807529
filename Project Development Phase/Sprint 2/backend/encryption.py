import hashlib

def makeHash(email):
    email = email + "hello"
    h = hashlib.sha256(email.encode())
    return (h.hexdigest())
