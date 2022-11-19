from flask import Flask, jsonify, request
import pymongo
from flask_cors import CORS
from sendEmail import sendEmail
from encryption import makeHash
from modelLoder import ValuePredictor
from dotenv import dotenv_values

config = dotenv_values(".env")


app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
conn_url = config["URL"]

client = pymongo.MongoClient(conn_url)


db = client["project0"]

usersCollection = db["users"]


@app.route("/auth/register", methods=["POST"])
def register():
    filteredProfile = usersCollection.find({"email": request.json["email"]})
    res = []
    for i in filteredProfile:
        res.append(i)
    if len(res) > 0:
        return {"status": False, "statusMessage": "Email already registered"}
    else:
        newUser = {"name": "", "email": "", "password": ""}
        newUser["name"] = request.json["name"]
        newUser["email"] = request.json["email"]
        newUser["password"] = request.json["password"]
        newUser["verifyStatus"] = False
        insertedProfile = usersCollection.insert_one(newUser)
        sendEmail(request.json["email"])

        return {"status": True, "verify": False}


@app.route("/auth/login", methods=["POST"])
def login():
    filteredProfile = usersCollection.find({"email": request.json["email"], "password": request.json["password"]})
    res = []
    for i in filteredProfile:
        res.append(i)

    if len(res) > 0:
        if res[0]["verifyStatus"] == False:
            return {"status": False, "statusMessage": "Please verify, link has been sended to your email"}
        return {"status": True}

    else:
        return {"status": False, "statusMessage": "No email found"}


@app.route("/auth/verify", methods=["POST"])
def verify():

    tempHash = makeHash(request.json["email"])
    if tempHash == request.json["hash"]:
        updatedInfo = usersCollection.update_one({"email": request.json["email"]}, {"$set": {"verifyStatus": True}})
        # print(updatedInfo)
        return {"status": True}
    else:
        return {"status": False, "statusMessage": "No email found"}


@app.route("/result", methods=["POST"])
def result():
    if request.method == "POST":
        to_predict_list = request.json
        print(to_predict_list)
        if len(to_predict_list) != 6:
            return {"status": False, "statusMessage": "All values are required"}
        to_predict_list = [
            to_predict_list["con"],
            to_predict_list["bod"],
            to_predict_list["nn"],
            to_predict_list["tc"],
            to_predict_list["ph"],
            to_predict_list["do"],
        ]

        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)

    return {"status": True, "calresult": result[0]}


@app.route("/")
def hello():
    return "ok"


if __name__ == "__main__":
    app.run(port=5000, Debug=False, Testing=False)
