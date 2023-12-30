import pytube as pt


class DownloadVideo:
    def __init__(self, url):
        self.url = url

    def download(self):
        yt = pt.YouTube(self.url)
        stream = yt.streams.filter(only_audio=True)[0]
        stream.download(filename="audio/audio1.mp3")
