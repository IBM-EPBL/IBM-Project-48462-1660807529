import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import "./App.css";
import Authentication from "./screen/authentication/authentication";
import Homescreen from "./screen/HomeScreen/homescreen";

function App() {
  return (
    <Routes>
      <Route path="/" element={<Authentication />} />
      <Route path="/home" element={<Homescreen />} />
    </Routes>
  );
}

export default App;
