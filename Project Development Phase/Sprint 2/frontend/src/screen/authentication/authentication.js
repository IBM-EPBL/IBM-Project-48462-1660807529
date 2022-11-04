import React, { useState } from "react";
import Login from "../../components/Login/Login.js";
import Register from "../../components/Register/Register.js";
import { useNavigate } from "react-router-dom";
import "./authentication.css";
import axios from "axios";

const Authentication = () => {
  const navigate = useNavigate();
  const [typeOfAuth, setTypeOfAuth] = useState("login");
  const [errorMessage, seterrorMessage] = useState(null);

  const onErrorHandler = (data) => {
    seterrorMessage(data);

    setTimeout(() => {
      seterrorMessage(null);
    }, [5000]);
  };

  const onLoginHandler = async (loginData) => {
    await axios
      .post("http://localhost:5000/auth/login", {
        email: loginData.email,
        password: loginData.password,
      })
      .then(async (data) => {
        // console.log(data);
        if (!data.data.status) {
          onErrorHandler(data.data.statusMessage);
        } else {
          sessionStorage.setItem("email", loginData.email);
          sessionStorage.setItem("name", data.data.name);
          sessionStorage.setItem("id", data.data.id);
          navigate("/home");
        }
      })
      .catch((err) => {
        console.log(err);
      });
  };
  const onRegisterHandler = async (Rdata) => {
    await axios
      .post("http://localhost:5000/auth/register", {
        email: Rdata.email.toLowerCase(),
        password: Rdata.password,
        name: Rdata.name,
      })
      .then(async (data) => {
        let s = data.data.status;
        if (data.data.status === false) {
          console.log(s);
          onErrorHandler(data.data.statusMessage);
        } else {
          console.log(s);
          console.log(data);
          sessionStorage.setItem("email", Rdata.email.toLowerCase());
          sessionStorage.setItem("name", data.data.name);
          sessionStorage.setItem("id", data.data.id);
          onErrorHandler(
            "Please verify your account , link has been sended to the mail"
          );
          
          setTypeOfAuth("login");
          // window.location.reload();
        }
      })
      .catch((err) => {
        console.log(err);
      });
  };

  return (
    <div className="outerScreen">
      <div className="content">
        <h1 className="greetingh1">Welcome to Hydropure</h1>
        {typeOfAuth === "login" ? (
          <Login
            callBack={(data) => {
              onLoginHandler(data);
            }}
            errorCallBack={(data) => {
              onErrorHandler(data);
            }}
          />
        ) : (
          <Register
            callBack={(data) => {
              onRegisterHandler(data);
            }}
            errorCallBack={(data) => {
              onErrorHandler(data);
            }}
          />
        )}
        <p
          className="formswitcherp"
          onClick={() => {
            setTypeOfAuth(typeOfAuth === "login" ? "signin" : "login");
          }}
        >
          {typeOfAuth === "login"
            ? "Don't you have account? Click Me"
            : "Do you have an account? click me"}
        </p>
        {errorMessage !== null && (
          <h1 className="errorMessageH1">{`${errorMessage}`}</h1>
        )}
      </div>
    </div>
  );
};

export default Authentication;
