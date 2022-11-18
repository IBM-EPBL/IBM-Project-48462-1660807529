import smtplib
from encryption import makeHash
from dotenv import dotenv_values

config = dotenv_values(".env")


def sendEmail(email):
    gmail_user = config["USER"]
    gmail_password = config["PASSWORD"]

    sent_from = gmail_user
    to = [email]
    subject = "Account verification"
    body = (
        "Greetings \n click this link to check verify your account \n\n https://hydropure-frontend.vercel.app/verify/email/"
        + email
        + "/"
        + makeHash(email)
    )

    email_text = """\
    From: %s
    To: %s
    Subject: %s

    %s
    """ % (
        sent_from,
        ", ".join(to),
        subject,
        body,
    )

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()

        return ("Email sent!", email_text)
    except:
        return "Something went wrong..."
