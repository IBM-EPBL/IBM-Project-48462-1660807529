import React, { useState } from "react";
import About from "../../components/about/about";
import Dashboard from "../../components/dashboard/dashboard";
import "./homescreen.css";

const Homescreen = () => {
  const [isActive, setActive] = useState("false");
  const ToggleClass = () => {
    setActive(!isActive);
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
          <button type="button">About</button>
          <button type="button" className="active">
            Calculator
          </button>
          <button type="button">states</button>
        </nav>
      </nav>
      <About />
      <Dashboard />
    </div>
  );
};

export default Homescreen;
