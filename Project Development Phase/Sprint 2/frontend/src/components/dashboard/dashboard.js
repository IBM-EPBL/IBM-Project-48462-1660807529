import React from "react";
import "./dashboard.css";
const Dashboard = () => {
  return (
    <div className="ds-outer">
      <h1>WQI Calculator:</h1>
      <div className="form">
        <div className="inputForm">
          <label for="text-input">CONDUCTIVITY (µmhos/cm):</label>
          <input className="input" id="text-input" type="number" />
        </div>
        <div className="inputForm">
          <label for="text-input">NITRATENAN N+ NITRITENANN (mg/l):</label>
          <input className="input" id="text-input" type="number" />
        </div>
        <div className="inputForm">
          <label for="text-input">POTENTIAL OF HYDROGEN (pH):</label>
          <input className="input" id="text-input" type="number" />
        </div>
        <div className="inputForm">
          <label for="text-input">DISSOLVED OXYGEN (mg/l):</label>
          <input className="input" id="text-input" type="number" />
        </div>
        <div className="inputForm">
          <label for="text-input">BIOCHEMICAL OXYGEN DEMAND (mg/l):</label>
          <input className="input" id="text-input" type="number" />
        </div>
        <div className="inputForm">
          <label for="text-input">TOTAL COLIFORM (MPN/100ml)Mean:</label>
          <input className="input" id="text-input" type="number" />
        </div>
      </div>
      <button className="ds-button">submit</button>
    </div>
  );
};

export default Dashboard;
