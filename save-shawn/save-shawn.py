from PTL import image
import pytessaract
import cv2
import youtube_dl

def get_video(url: str) -> None:
    ydl_opts = {}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

def process_frame(frame) -> None:
    pass

def parse(filepath) -> None:
    pass
