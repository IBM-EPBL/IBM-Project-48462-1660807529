import React from "react";
import "./about.css";
const About = () => {
  return (
    <div id="about" className="about-outer">
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
        <ul>
          <li>
            Excellent: (WQI Value 95-100) – Water quality is protected with a
            virtual absence of impairment; conditions are very close to pristine
            levels. These index values can only be obtained if all measurements
            meet recommended guidelines virtually all of the time.
          </li>
          <li>
            Very Good: (WQI Value 89-94) – Water quality is protected with a
            slight presence of impairment; conditions are close to pristine
            levels.
          </li>
          <li>
            Good: (WQI Value 80-88) – Water quality is protected with only a
            minor degree of impairment; conditions rarely depart from desirable
            levels.
          </li>
          <li>
            Fair: (WQI Value 65-79) – Water quality is usually protected but
            occasionally impaired; conditions sometimes depart from desirable
            levels.
          </li>
          <li>
            Marginal: (WQI Value 45-64) – Water quality is frequently impaired;
            conditions often depart from desirable levels.
          </li>
          <li>
            Poor: (WQI Value 0-44) – Water quality is almost always impaired;
            conditions usually depart from desirable levels
          </li>
        </ul>
      </p>
    </div>
  );
};

export default About;
