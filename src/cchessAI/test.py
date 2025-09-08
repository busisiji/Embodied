import requests

session = requests.Session()
headers = {
    "User-Agent": "Mozilla/5.0"
}

login_url = "http://www.dpxq.com/hldcg/search/login.asp"
data = {
    "username": "busisiji",
    "password": "zwc6731061"
}

response = session.post(login_url, data=data, headers=headers)
print(response.status_code)
print(response.text)
