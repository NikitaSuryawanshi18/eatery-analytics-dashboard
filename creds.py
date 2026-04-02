from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()
#https://curule-fumblingly-sean.ngrok-free.dev
CLIENT_ID = "sq0idp-yMdKyiJUPKd1p239Fm2qcA"
CLIENT_SECRET = "sq0csp-pUTyuYgdsdOGSb6U8M4GzrXJeo8fxPfbgbolNZrzp6M"

@app.get("/")
def home():
    return {"status": "server running"}

@app.get("/callback")
def callback(request: Request):
    code = request.query_params.get("code")

    if not code:
        return {"error": "No code received"}

    # Exchange code for token
    url = "https://connect.squareup.com/oauth2/token"

    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code"
    }

    response = requests.post(url, json=payload)

    data = response.json()

    return {
        "message": "Connected successfully",
        "data": data
    }