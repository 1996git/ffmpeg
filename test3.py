import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, filedialog
from moviepy.editor import VideoFileClip, concatenate_videoclips
import librosa
import os
from PIL import Image, ImageTk
import threading
import time

selected_video_paths = []

class VideoAnalyzer:
    def __init__(self, parent, row, column):
        self.parent = parent
        self.row = row
        self.column = column
        self.video_path = ""

        # キャンバスの作成
        self.canvas = tk.Canvas(parent, bg="black", width=400, height=300)
        self.canvas.grid(row=row*2, column=column, padx=10, pady=10)

        # ファイル選択ボタンの作成
        self.select_button = ttk.Button(parent, text="Select Video", command=self.select_video)
        self.select_button.grid(row=row *2 + 2, column=column, pady=5)
        self.input_text = tk.StringVar(value="")  # テキストボックスの初期値を空文字列に設定

        # テキストボックスの作成
        self.input_entry = ttk.Entry(parent, textvariable=self.input_text)
        self.input_entry.grid(row=row*2+1, column=column, pady=5)
        # オーディオ波形をプロットするキャンバス
        self.plot_canvas = None

        # キャプチャを保持する変数
        self.capture = None
        
    def get_input_text(self):
        return self.input_text.get()

    def select_video(self):
        global selected_video_paths
        self.video_path = filedialog.askopenfilename()
        selected_video_paths.append(self.video_path)
        if self.video_path:
            self.plot_audio_waveform()
            video_thread = threading.Thread(target=self.play_video)
            video_thread.start()


    def plot_audio_waveform(self):
        try:
            # オーディオファイルを一時保存
            video_clip = VideoFileClip(self.video_path)
            audio_clip = video_clip.audio
            audio_output_path = f"./output_audio_{self.row}{self.column}.wav"
            audio_clip.write_audiofile(audio_output_path, codec='pcm_s16le', ffmpeg_params=['-ac', '1'])

            # オーディオ波形を読み込んでプロット
            waveforms, sample_rate = librosa.load(audio_output_path, sr=None, mono=True)
            fig = plt.figure(figsize=(4, 2.4))
            ax = fig.add_subplot(111)
            ax.plot(waveforms, color='blue')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.set_title('Audio Waveform')
            self.plot_canvas = FigureCanvasTkAgg(fig, master=self.parent)
            self.plot_canvas.draw()
            self.plot_canvas.get_tk_widget().grid(row=self.row*2, column=self.column + 1, rowspan=2, padx=10, pady=5)
        except Exception as e:
            print("エラーが発生しました:", e)

    def play_video(self):
        try:
            # 動画を再生
            self.capture = cv2.VideoCapture(self.video_path)
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            aspect_ratio = width / height
            canvas_width = 400
            canvas_height = int(canvas_width / aspect_ratio)
            self.canvas.config(width=canvas_width, height=canvas_height)

            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (canvas_width, canvas_height))
                self.canvas.delete("all")
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.parent.update()
                time.sleep(1 / 30)  
        except Exception as e:
            print("エラーが発生しました:", e)
        finally:
            if self.capture:
                self.capture.release()

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def merge_videos(video_analyzers,duration,resolution):
    global selected_video_paths
    if len(selected_video_paths) >= 1:
        print("Merging videos...")
        caps = [cv2.VideoCapture(video_path) for video_path in selected_video_paths]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (FRAME_WIDTH * 2, FRAME_HEIGHT * 2))

        start_time = time.time()
        end_time = start_time + duration
        labels = [video_analyzer.get_input_text() for video_analyzer in video_analyzers]

        # 動画の再生と保存
        while time.time()<end_time:
            frames = []
            for i, cap in enumerate(caps):
                ret, frame = cap.read()
                if not ret:
                    print(f"Video {i+1} has finished playing.")
                    break
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

                # ラベルをフレームに追加
                label = labels[i]
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                frames.append(frame)

            if len(frames) == 1:
                output_frame = frames[0]
            elif len(frames) == 2:
                output_frame = np.hstack(frames)
            elif len(frames) == 3:
                output_frame = np.vstack([np.hstack(frames[:2]), frames[2]])
            elif len(frames) == 4:
                frame1, frame2, frame3, frame4 = frames
                output_frame = np.vstack([np.hstack([frame1, frame2]), np.hstack([frame3, frame4])])

            output_video.write(output_frame)
            resized_frame = cv2.resize(output_frame,resolution)
            cv2.imshow("Video", output_frame)

            # ウィンドウを閉じるボタンが押された場合にウィンドウを閉じる
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # ウィンドウが閉じられた場合にループを終了する
            if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(1 / 33)


 #Tkinterウィンドウの作成
       # window = tk.Tk()
       # window.title("Video Analyzer")

 #動画分析器の作成
#video_analyzers = []# for i in range(2):
#for j in range(2):
         #video_analyzer = VideoAnalyzer(window, i*2 + 1, j*2)
         #video_analyzers.append(video_analyzer)

         #select_button = ttk.Button(window, text="Merge Video", command=lambda: merge_videos(video_analyzers))
         #select_button.grid(row=0, column=0, pady=0)

# 以下は第4のタブのコードです
         #tab_control = ttk.Notebook(window)
         #fourth_tab = ttk.Frame(tab_control)
         #fourth_tab.pack(expand=True, fill="both")
        # tab_control.add(fourth_tab, text="merge")

# ここに必要なコードを追加してください

# window.mainloop()

