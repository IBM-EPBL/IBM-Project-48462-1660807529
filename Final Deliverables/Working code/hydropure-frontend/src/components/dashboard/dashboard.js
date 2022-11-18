import React, { useState, useEffect } from "react";
import "./dashboard.css";
const Dashboard = (props) => {
  const [input, setInput] = useState({
    con: 0,
    nn: 0,
    ph: 0,
    do: 0,
    bod: 0,
    tc: 0,
  });
  const [status, setstatus] = useState("--");
  useEffect(() => {
    if (props.res >= 95) {
      setstatus("Excellent");
    } else if (94 >= props.res && props.res >= 89) {
      setstatus("Very Good");
    } else if (80 <= props.res && props.res <= 88) {
      setstatus("Good");
    } else if (65 <= props.res && props.res <= 79) {
      setstatus("Fair");
    } else if (45 <= props.res && props.res <= 64) {
      setstatus("Marginal");
    } else if (0 <= props.res && props.res <= 44) {
      setstatus("Poor");
    }
  }, [props.res]);

  const onChangeHandler = (data) => {
    // console.log(input);
    setInput({ ...input, ...data });
    // console.log(input);
  };

  return (
    <div className="ds-outer">
      <h1>WQI Calculator:</h1>
      <div className="form">
        <div className="inputForm">
          <label for="text-input">CONDUCTIVITY (µmhos/cm):</label>
          <input
            min={0}
            max={100}
            className="input"
            id="text-input"
            onChange={(e) => {
              onChangeHandler({ con: e.target.value });
            }}
            type="number"
          />
        </div>
        <div className="inputForm">
          <label for="text-input">NITRATENAN N+ NITRITENANN (mg/l):</label>
          <input
            min={0}
            max={100}
            className="input"
            id="text-input"
            type="number"
            onChange={(e) => {
              onChangeHandler({ nn: e.target.value });
            }}
          />
        </div>
        <div className="inputForm">
          <label for="text-input">POTENTIAL OF HYDROGEN (pH):</label>
          <input
            min={0}
            max={100}
            className="input"
            id="text-input"
            type="number"
            onChange={(e) => {
              onChangeHandler({ ph: e.target.value });
            }}
          />
        </div>
        <div className="inputForm">
          <label for="text-input">DISSOLVED OXYGEN (mg/l):</label>
          <input
            min={0}
            max={100}
            className="input"
            id="text-input"
            type="number"
            onChange={(e) => {
              onChangeHandler({ do: e.target.value });
            }}
          />
        </div>
        <div className="inputForm">
          <label for="text-input">BIOCHEMICAL OXYGEN DEMAND (mg/l):</label>
          <input
            min={0}
            max={100}
            className="input"
            id="text-input"
            type="number"
            onChange={(e) => {
              onChangeHandler({ bod: e.target.value });
            }}
          />
        </div>
        <div className="inputForm">
          <label for="text-input">TOTAL COLIFORM (MPN/100ml)Mean:</label>
          <input
            min={0}
            max={100}
            className="input"
            id="text-input"
            type="number"
            onChange={(e) => {
              onChangeHandler({ tc: e.target.value });
            }}
          />
        </div>
      </div>
      <button
        className="ds-button"
        onClick={() => props.onSubmitHandler(input)}
      >
        submit
      </button>
      <h3 className="rs-h1">
        Water Quality Index : {parseFloat(props.res).toFixed(2)}
      </h3>
      <h1 className="rs-h1">Quality : {status}</h1>
    </div>
  );
};

export default Dashboard;
