import sounddevice as sd
import numpy as np
import queue
import threading
from faster_whisper import WhisperModel


class SpeechToText:
    def __init__(self, exercises, model_size="base", sample_rate=16000, chunk_seconds=2):
        self.exercises = exercises
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_seconds)
        self.audio_queue = queue.Queue()
        self.transcript_lock = threading.Lock()
        self.full_transcript = ""

        self.model = WhisperModel(model_size, device="cpu", compute_type="float32")

        self.stream = None
        self.running = False
        self.worker_thread = None

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(status)
        self.audio_queue.put(indata.copy())

    def _transcription_worker(self):
        buffer = np.zeros((0, 1), dtype=np.float32)

        while self.running:
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                buffer = np.concatenate((buffer, chunk))

                if len(buffer) >= self.chunk_size:
                    audio = buffer[:self.chunk_size].flatten()
                    buffer = buffer[self.chunk_size:]

                    segments, _ = self.model.transcribe(
                        audio,
                        beam_size=1,
                        language="en",
                        initial_prompt=f"Fitness exercise session. Terms include {" , ".join(self.exercises)}. The speaker may be counting repetitions and naming exercises.",
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=300),
                    )

                    text = " ".join(seg.text.strip() for seg in segments)

                    if text:
                        with self.transcript_lock:
                            self.full_transcript += " " + text

            except queue.Empty:
                continue

    def start_stream(self):
        if self.running:
            return

        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=0,
        )
        self.stream.start()

        self.worker_thread = threading.Thread(target=self._transcription_worker, daemon=True)
        self.worker_thread.start()

    def clear_transcript(self):
        self.full_transcript = ""

    def stop_stream(self):
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()

    def get_transcription(self):
        with self.transcript_lock:
            return self.full_transcript.strip()