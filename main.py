import openai
import torch
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification, AutoConfig
import librosa
from moviepy.editor import VideoFileClip
import os


def convert_video_to_audio_moviepy(video_file, output_ext="mp3"):
    """Converts video to audio using MoviePy library
    that uses `ffmpeg` under the hood"""
    filename, ext = os.path.splitext(video_file)
    clip = VideoFileClip(video_file)
    clip.audio.write_audiofile(f"{filename}.{output_ext}")

# ######################
# Openai setup
openai.organization = "org-W1xrGR4WAmmdeqk5vOBvlntj"
openai.api_key = "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"
# openai.Model.list()

# ######################
# Prompting ChatGPT
# "gpt-3.5-turbo"
# response = openai.ChatCompletion.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are an assistant to an autistic person"},
#         {"role": "user", "content": "A person approached me. He appears neutral and slighly irritated. How should I act? Please respond with a single sentence."},
#         {"role": "assistant", "content": "You should speak to him mindfully and assess the situtaion"},
#         {"role": "user", "content": "His expression changed: He appears angrier. His tone changed: he sounds disgusted. How should I act? Please respond with a single sentence."}
#     ]
# )
#
# print(response["choices"][0]['message']['content'])

# ######################
# Converting video to audio
# convert_video_to_audio_moviepy(r"C:\Users\sharp\PycharmProjects\Hackathon2023\Vid2.mp4")

# ######################
# Transcribe audio
audio = open(r'long_dialog.wav', 'rb')
transcript = openai.Audio.transcribe('whisper-1', audio, language='en', response_format="verbose_json")
print(transcript)
text = transcript["text"]
print(text)
audio.close()

# ######################
# Classify audio emotion
# processor = Wav2Vec2FeatureExtractor.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# config = AutoConfig.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition")
# model = AutoModelForAudioClassification.from_pretrained("ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition", config=config)
# model.eval()
# #
# audio, _ = librosa.load(r"audio/Vid.mp3")
# inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
# print(inputs.input_values.shape)
# #
# audio_to_emotion_dict = {0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised'}
# outputs = model(inputs.input_values, attention_mask=inputs.attention_mask)
# # print(outputs.logits)
# print(audio_to_emotion_dict[torch.argmax(outputs.logits).item()])

# from typing import Dict, List, Tuple
# from dataclasses import dataclass
# @dataclass
# class EmotionsCheckpoint:
#     emotions_slices_dict: Dict[int, List[Tuple[int]]]
#     """
#     Example:
#     emotions_slices_dict = {0: [3,4,5, 7, 9,10,11],
#                             1: [],
#                             2: [(6,19)]
#                             ...
#                             }
#     """
#
#     def fill_holes(self):
#         for emotion, idxs in self.emotions_slices_dict.items():
#             last_idx = -3
#             new_list = []
#             for idx in idxs:
#                 if idx - last_idx == 2:
#                     new_list.append(idx - 1)
#                 new_list.append(idx)
#                 last_idx = idx
#             self.emotions_slices_dict[emotion] = new_list
#
# a = EmotionsCheckpoint({
#     1: [1,3],
#     2: [],
#     3: [2,3,5,7,9,12],
#     4: [2]
# })
# a.emotions_slices_dict = {
#     1: [1,3],
#     2: [],
#     3: [2,3,5,7,9,12],
#     4: [2]
# }
# a.fill_holes()
# print(a.emotions_slices_dict)


