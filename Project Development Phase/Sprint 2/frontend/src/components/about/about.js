import React from "react";
import "./about.css";
const About = () => {
  return (
    <div className="about-outer">
      <h1>HydroPure - Efficient Water Quality Analysis and Prediction</h1>
      <p>
        <h2>Welcome to HydroPure!</h2> Using the book, Field Manual for Water
        Quality Monitoring, the National Sanitation Foundation surveyed 142
        people, representing a wide range of positions at the local, state, and
        national level, about 35 water quality tests for possible inclusion in
        an index.Among which six factors were chosen for WQI calculation. The
        Water Quality Index is a 100-point scale that summarizes results from a
        total of six different measurements when complete.This website takes in
        six parameters as input and calculates the WQI. The parameters
        considered are:
        <ul>
          <li>Conductivity</li>
          <li>Nitratenan N + Nitrite Ann</li>
          <li>pH</li>
          <li>Dissolved Oxygen</li>
          <li>Biochemical oxygen demand</li>
          <li>Total Coliform mean</li>
        </ul>
        This website allows the user to check the water quality and the purpose
        it can be used for, by accepting the above mentioned 6 inputs and
        displaying the Water Quality Index(wqi) as output.
      </p>
    </div>
  );
};

export default About;
