import os
import json
from typing import Dict, List, Optional
import cv2
import pytesseract
from PIL import Image
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import yt_dlp
from datetime import timedelta
import sys
import logging

# API keys
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
log = logging.getLogger(__name__)

class ShawnException(Exception):
    pass

class VideoProcessor:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.text_timestamps: Dict[str, float] = {}

    def get_video(self, url: str) -> str:
        ydl_opts = {
            'format': 'best',
            'outtmpl': 'assets/video.%(ext)s'
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return ydl.prepare_filename(info)

    def preprocess_frame(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return thresh


    def process_frame(self, frame: np.ndarray) -> Optional[str]:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        try:
            text = pytesseract.image_to_string(image=pil_image, lang="chi_sim")
            text = text.strip()
            return text if text else None
        except ShawnException as e:
            print(f"Error processing frame: {e}")
            return None

    def translate_text_batch(self, text_blocks: List[Dict[str, str]]) -> List[Dict[str, str | float]]:
        if not text_blocks:
            return []

        prompt = "Please translate the following phrases from Chinese to English:\n"
        max_tokens = 4096
        batched_prompts = []
        batch = prompt

        for entry in text_blocks:
            if len(batch) + len(entry["text"]) > max_tokens:
                batched_prompts.append(batch)
                batch = prompt
            batch += entry["text"] + "\n"

        if batch != prompt:
            batched_prompts.append(batch)

        translations = []
        for prompt in batched_prompts:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a Chinese to English translator. Translate any Chinese characters you find, even if the text seems incomplete or mixed with other characters. Only output the translation, no explanations."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )

                if response.choices[0].message.content is None:
                    raise ShawnException("Empty response from gpt")

                translation_text = response.choices[0].message.content.strip()
                translations_list = [line.strip() for line in translation_text.split('\n') if line.strip()]

                for i, translation in enumerate(translations_list):
                    if i < len(text_blocks):
                        translations.append({
                            "timestamp": text_blocks[i]["timestamp"],
                            "text": text_blocks[i]["text"],
                            "translation": translation
                        })
            except Exception as e:
                log.error(f"Translation error: {e}")

        return translations


    def process_video(self, video_path: str, output_path: str = "translations.json") -> None:
        log.info(f"[process_video]: ocr on {video_path} into {output_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        text_buffer = []
        frame_count = 0

        timestamp_int = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_count / fps
            frame_count += 1

            if timestamp > timestamp_int + 1:
                timestamp_int += 1
            else:
                continue

            text = self.process_frame(frame)

            if text and text not in self.text_timestamps:
                log.info(f"[process_video]: grabbed unique text at timestamp {timestamp}: `{text}`")
                self.text_timestamps[text] = timestamp
                text_buffer.append({
                    "timestamp": str(timedelta(seconds=timestamp)),
                    "text": text
                })

                if len(text_buffer) >= 5:
                    log.info(f"batch processing: {text_buffer}")
                    translations = self.translate_text_batch(text_buffer)
                    self.save_translations(translations, output_path)
                    text_buffer = []

        if text_buffer:
            translations = self.translate_text_batch(text_buffer)
            self.save_translations(translations, output_path)
            # self.save_translations(text_buffer, output_path)

        cap.release()

    def save_translations(self, translations: List[Dict[str, str | float]], output_path: str) -> None:
        log.info(f"[save_translations]: writing to {output_path}")
        try:
            existing_translations = []
            if os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_translations = json.load(f)

            existing_translations.extend(translations)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(existing_translations, f, ensure_ascii=False, indent=2)
        except ShawnException as e:
            print(f"Error saving translations: {e}")

def main(url: str):
    processor = VideoProcessor()
    video_path = processor.get_video(url)
    processor.process_video(video_path)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3.12 save-shawn.py <youtube_url>")
        sys.exit(1)
    main(sys.argv[1])
