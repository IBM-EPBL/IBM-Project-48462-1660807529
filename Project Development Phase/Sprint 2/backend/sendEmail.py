import smtplib
from encryption import makeHash;

def sendEmail(email):
    gmail_user = '1905122cse@cit.edu.in'
    gmail_password = 'Svs@2001'
    
    sent_from = gmail_user
    to = ["prembalaraman056@gmail.com"] 
    subject = 'Account verification'
    body = 'Greetings \n click this link to check verify your account \n\n http://localhost:3000/verify/email/'+email+"/"+makeHash(email)

    email_text = """\
    From: %s
    To: %s
    Subject: %s

    %s
    """ % (sent_from, ", ".join(to), subject, body)

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.ehlo()
        server.login(gmail_user, gmail_password)
        server.sendmail(sent_from, to, email_text)
        server.close()

        print('Email sent!',email_text)
    except:
        print('Something went wrong...')
