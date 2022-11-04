import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import axios from "axios";

const VerificationScreen = (props) => {
  const [Status, setStatus] = useState("wait");
  const [errorMessage, seterrorMessage] = useState(null);
  const key = useParams();

  const onErrorHandler = (data) => {
    seterrorMessage(data);

    setTimeout(() => {
      seterrorMessage(null);
    }, [5000]);
  };
  useEffect(() => {
    sendVerifiyableCode();
  });
  const sendVerifiyableCode = async () => {
    await axios
      .post("http://localhost:5000/auth/verify", {
        email: key["email"],
        hash: key["hash"],
      })
      .then(async (data) => {
        if (!data.data.status) {
          setStatus("not");
        } else {
          setStatus("verified");
        }
      })
      .catch((err) => {
        console.log(err);
      });
  };
  if (Status === "wait") {
    return <div>{"loading..."}</div>;
  }
  if (Status === "verified") {
    return (
      <div>
        Account verified , Try <a href="http://localhost:3000/">login</a>
      </div>
    );
  }
  if (Status === "not") {
    return <div>Invalid Url</div>;
  }
};

export default VerificationScreen;
