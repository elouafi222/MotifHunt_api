import flask
from flask import jsonify, request
from transcripteur import transcription
from sumarizer import summarazire
from downloadVideo import DownloadVideo
from flask_cors import CORS
import os

app = flask.Flask(__name__)
CORS(app)


@app.route("/text2", methods=["GET"])
def get_text2():
    #### à changer a une classe pour une bonne vitesse
    url1 = "https://www.youtube-nocookie.com/embed/5BeOh1XHAVI"
    url2 = "https://www.youtube.com/watch?v=bJzb-RuUcMU"
    donwobj = DownloadVideo(url1)
    donwobj.download()

    #####################################################
    # pathAudio = os.path.join("audio", "audio1.mp3")
    objtranscripteur = transcription(nomAudio="audio/audio1.mp3")
    text = objtranscripteur.execute()
    objsu = summarazire()
    texteresumer = objsu.summarize_text(text)

    return jsonify({"text": text, "textSummarize": texteresumer})


@app.route("/text", methods=["GET", "POST"])
def get_text():
    if request.method == "POST":
        # Récupérer les données du corps de la requête POST
        data = request.get_json()
        Url = data["valueFromChild"]
        #### à changer a une classe pour une bonne vitesse
        donwobj = DownloadVideo(Url)
        donwobj.download()

        #####################################################
        # pathAudio = os.path.join("audio", "audio1.mp3")
        objtranscripteur = transcription(nomAudio="audio/audio1.mp3")
        text = objtranscripteur.execute()
        objsu = summarazire()
        texteresumer = objsu.summarize_text(text)

        # Retournez la réponse en format JSON
        print(Url)
        return jsonify({"textReceived": texteresumer, "textSummarize": texteresumer})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

    """
    
    """
