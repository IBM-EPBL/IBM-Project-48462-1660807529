import React, { useState } from "react";
import About from "../../components/about/about";
import Dashboard from "../../components/dashboard/dashboard";
import "./homescreen.css";
import axios from "axios";

const Homescreen = () => {
  const [isActive, setActive] = useState("false");
  const [res, setRes] = useState("--");
  const ToggleClass = () => {
    setActive(!isActive);
  };
  const [status, setstatus] = useState("--");
  const [errorMessage, seterrorMessage] = useState(null);

  const onErrorHandler = (data) => {
    seterrorMessage(data);

    setTimeout(() => {
      seterrorMessage(null);
    }, [5000]);
  };

  const calculator = async (data) => {
    // console.log(data);
    await axios
      .post("http://127.0.0.1:5000/result", data)
      .then(async (data) => {
        if (data.data.status === false) {
          // // console.log(data.data.statusMessage);
          onErrorHandler(data.data.statusMessage);
        } else {
          setRes(data.data.calresult);
        }
      })
      .catch((err) => {
        // console.log(err);
      });
  };

  return (
    <div className={isActive ? "hs-outer" : "hs-outer open"}>
      <nav className="navbar">
        <div className="navbar-overlay" onClick={ToggleClass}></div>
        <button type="button" className="navbar-burger" onClick={ToggleClass}>
          <span className="material-icons">i</span>
        </button>
        <h1 className="navbar-title">HydroPure</h1>
        <nav className="navbar-menu">
          <a href="#about">
            <button type="button" className="active">
              About
            </button>
          </a>
          <a href="#calculator">
            <button type="button" className="active">
              Calculator
            </button>
          </a>
        </nav>
      </nav>
      <section id="about">
        <About />
      </section>
      <section id="calculator">
        <Dashboard
          onSubmitHandler={(data) => {
            calculator(data);
            // calculator(data);
          }}
          res={res}
        />
      </section>
      {errorMessage !== null && (
        <h1 className="errorMessageH1">{`${errorMessage}`}</h1>
      )}
    </div>
  );
};

export default Homescreen;
