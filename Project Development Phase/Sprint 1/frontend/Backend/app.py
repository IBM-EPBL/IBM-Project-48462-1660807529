from flask import Flask,jsonify,request
import pymongo
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/*" : {"origins": "*"}})

conn_url = "mongodb+srv://batch5:batch5@cluster0.dblogvz.mongodb.net/test"

try:
    client = pymongo.MongoClient(conn_url)
except Exception:
    print("[-] Unable to connect to the database.")


db = client["project0"]

usersCollection = db["users"]

@app.route("/auth/register",methods=['POST'])
def register():
    filteredProfile = usersCollection.find({"email":request.json["email"]})
    res = []
    for i in filteredProfile:
        res.append(i)
    if(len(res)>0):
        return {"status":False,"statusMessage":"Email already registered"}
    else:
        newUser = {"name":"","email":"","password":""}
        newUser["name"] = request.json["name"]
        newUser["email"] = request.json["email"]
        newUser["password"] = request.json["password"]
        insertedProfile = usersCollection.insert_one(newUser)
        print(insertedProfile)
        return {"status":True}


@app.route("/auth/login",methods=['POST'])
def login():
    filteredProfile = usersCollection.find({"email":request.json["email"] , "password":request.json["password"]}) 
    res = []
    for i in filteredProfile:
        res.append(i)
    if(len(res)>0):
        return {"status":True}
    else:
        return {"status":False,"statusMessage":"No email found"}
    

@app.route('/')
def hello():
    print("hello")
    return ("ok")

if __name__ == '__main__':
    app.run(port=5000,Debug=False,Testing=False)


