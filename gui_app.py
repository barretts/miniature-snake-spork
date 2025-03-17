import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
import torchaudio
import torchaudio.transforms as T
import torch
import numpy as np
from scipy.io import wavfile
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# ----------------------------------------------------------------------------
# start playing with imports -------------------------------------------------
# ----------------------------------------------------------------------------
import importlib.util

# Define the path to the CrisperWhisper repository
crisperwhisper_path = os.path.abspath("CrisperWhisper")
utils_path = os.path.join(crisperwhisper_path, "utils.py")

# Ensure the path exists before importing
if not os.path.exists(utils_path):
    raise ImportError(f"utils.py not found in {crisperwhisper_path}")

# Dynamically load utils.py under the custom namespace "cw_utils"
spec = importlib.util.spec_from_file_location("cw_utils", utils_path)
cw_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cw_utils)

# Now you can use cw_utils.adjust_pauses_for_hf_pipeline_output
adjust_pauses_for_hf_pipeline_output = cw_utils.adjust_pauses_for_hf_pipeline_output
# ----------------------------------------------------------------------------
# end playing with imports ---------------------------------------------------
# ----------------------------------------------------------------------------

def load_audio_file(file_path):
    """Load any supported audio format (MP3, WAV, OGG) and resample to 16kHz for model compatibility."""
    waveform, sample_rate = torchaudio.load(file_path)  # Automatically detects the format
    
    # Convert to mono (if the file has multiple channels)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz for CrispWhisper compatibility
    if sample_rate != 16000:
        resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    return waveform.squeeze(0).numpy()  # Convert tensor to NumPy array

# Helper: convert seconds to SRT timestamp format "HH:MM:SS,mmm"
def seconds_to_srt_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# Convert transcription chunks to SRT format
def timestamps_to_srt(chunks):
    srt_str = ""
    for idx, word in enumerate(chunks, start=1):
        start, end = word["timestamp"]
        start_str = seconds_to_srt_timestamp(start)
        end_str = seconds_to_srt_timestamp(end)
        text = word["text"]
        srt_str += f"{idx}\n{start_str} --> {end_str}\n{text}\n\n"
    return srt_str

# The main GUI class using Tkinter
class CrispWhisperGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CrisperWhisper SRT Generator")
        self.file_path = None
        self.transcription = None

        self.create_widgets()
        self.setup_model()

    def create_widgets(self):
        self.select_button = tk.Button(self.root, text="Select Audio File", command=self.select_file)
        self.select_button.pack(pady=10)

        self.file_label = tk.Label(self.root, text="No file selected")
        self.file_label.pack(pady=5)

        self.transcribe_button = tk.Button(self.root, text="Transcribe", command=self.transcribe_audio)
        self.transcribe_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="")
        self.status_label.pack(pady=5)

        self.save_button = tk.Button(self.root, text="Save SRT File", command=self.save_srt, state=tk.DISABLED)
        self.save_button.pack(pady=10)

    def setup_model(self):
        self.status_label.config(text="Loading model...")
        self.root.update()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model_id = "nyrahealth/CrisperWhisper"  # same as in your transcribe.py :contentReference[oaicite:4]{index=4}
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps="word",
            torch_dtype=torch_dtype,
            device=device,
        )
        self.status_label.config(text="Model loaded.")

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.ogg")]
        )
        if self.file_path:
            self.file_label.config(text=os.path.basename(self.file_path))
        else:
            self.file_label.config(text="No file selected")

    def transcribe_audio(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select an audio file first.")
            return
        self.status_label.config(text="Transcribing...")
        self.transcribe_button.config(state=tk.DISABLED)
        threading.Thread(target=self.run_transcription).start()

    def run_transcription(self):
        try:
            waveform = load_audio_file(self.file_path)
            result = self.pipe(waveform, return_timestamps="word")
            # Adjust pause timings using your helper from utils.py
            result = adjust_pauses_for_hf_pipeline_output(result)
            self.transcription = result
            self.status_label.config(text="Transcription complete.")
            self.save_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
            self.status_label.config(text="Error during transcription.")
        finally:
            self.transcribe_button.config(state=tk.NORMAL)

    def save_srt(self):
        if not self.transcription:
            messagebox.showerror("Error", "No transcription available.")
            return
        srt_content = timestamps_to_srt(self.transcription["chunks"])
        save_path = filedialog.asksaveasfilename(
            title="Save SRT File",
            defaultextension=".srt",
            filetypes=[("SRT Files", "*.srt")]
        )
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(srt_content)
            messagebox.showinfo("Success", f"SRT file saved to {save_path}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CrispWhisperGUI(root)
    root.mainloop()
