import requests

URL = "http://127.0.0.1:5000/predict/{}"

# Exemple de features d'una senyal valors del CSV
examplen = [140.5625,55.68378214,-0.234571412,-0.699648398,3.199832776,19.11042633,7.975531794,74.24222492,0]  # primers 

payload = {"features": examplen}

for model in ["logistic", "svm", "tree", "knn"]:
    for i in range(2):
        resp = requests.post(URL.format(model), json=payload)
        print(model, i, resp.json())
