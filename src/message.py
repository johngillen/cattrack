import smtplib, ssl
import src.database as database
from src.config import config

def send(message):
    username = config['email']['username']
    password = config['email']['password']
    server = config['email']['server']
    port = config['email']['port']
    recipient = config['notify']
    
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(server, port, context=context) as server:
        server.login(username, password)
        server.sendmail(username, recipient, message)
