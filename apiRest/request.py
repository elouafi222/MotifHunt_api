import requests

url = "http://127.0.0.1:5000/text"  # Assurez-vous que le serveur est en cours d'exécution sur le bon port

# Vous pouvez ajuster les données que vous envoyez dans la requête POST
data = {"valueFromChild": "https://www.youtube.com/watch?v=bJzb-RuUcMU"}

# Envoi de la requête POST
response = requests.post(url, json=data)

# Affichage de la réponse
print(response.json())
