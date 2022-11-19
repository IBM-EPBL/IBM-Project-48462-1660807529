import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";
import Authentication from "./screen/authentication/authentication";
import Homescreen from "./screen/HomeScreen/homescreen";
import VerificationScreen from "./screen/verificationScreen/verificationScreen";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Authentication />} />
      <Route path="/home" element={<Homescreen />} />
      <Route path="/verify/email/:email/:hash" element={<VerificationScreen />} />
    </Routes>
  );
}

export default App;
