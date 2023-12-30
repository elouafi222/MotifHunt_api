import whisper
import torch
import os


# choco install ffmpeg
class transcription:
    "this class for trancripe audio to text"

    def __init__(
        self,
        nomAudio="audio.mp3",
        Namemodel="base",
    ):
        self.nomAudio = nomAudio
        if not os.path.exists(self.nomAudio):
            print(f"Le fichier audio {self.nomAudio} n'a pas été trouvé.")
        else:
            print("le fichier existe")
        self.model = whisper.load_model(Namemodel)

    def execute(self):
        print(self.nomAudio)
        result = self.model.transcribe(self.nomAudio)
        return result["text"]


"""
obj = transcription()
print(obj.execute())
"""
