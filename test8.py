import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog
from moviepy.editor import VideoFileClip
import threading
import time
import os
import librosa
from PIL import Image, ImageTk
import subprocess
import pandas as pd
import ffmpeg
import ctypes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your_service_account_key.json"

class VideoPlayer:
    def __init__(self, parent, width=800, height=500, video_width=600, video_height=500):
        self.parent = parent
        self.canvas = tk.Canvas(parent, bg="black", width=width, height=height)
        self.canvas.pack(expand=True, fill="both")
        self.video_width = video_width
        self.video_height = video_height
        self.video_x = (width - video_width) // 2 + 250  # 動画キャンバスのX座標
        self.video_y = (height - video_height) // 2  # 動画キャンバスのY座標
        self.video_path = None
        self.capture = None
        self.isPlaying = False  # 動画の再生状態を管理するフラグ
        self.canvas.bind("<Button-1>", lambda event: self.play_pause_video())  # 画面をクリックして再生/停止を切り替える
        self.canvas.focus_set()
        self.frame_count = 0

        # キーボードイベントのバインド
        self.canvas.bind_all("<Left>", lambda event: self.adjust_video_position(-1))
        self.canvas.bind_all("<Right>", lambda event: self.adjust_video_position(1))
        self.canvas.bind("<space>", lambda event: self.play_pause_video())

        # アイコンの読み込み
        self.play_icon = Image.open(r"C:\Users\yuki.shibashi\Desktop\aikon\aikon1.png")
        self.play_icon2 = Image.open(r"C:\Users\yuki.shibashi\Desktop\aikon\aikon2.png")
        self.play_icon3 = Image.open(r"C:\Users\yuki.shibashi\Desktop\aikon\aikon3.png")
        self.play_icon4 = Image.open(r"C:\Users\yuki.shibashi\Desktop\aikon\aikon4.png")
        self.play_icon = self.play_icon.resize((32, 32))
        self.play_icon2 = self.play_icon2.resize((32, 32))
        self.play_icon3 = self.play_icon3.resize((50, 50))
        self.play_icon4 = self.play_icon4.resize((32, 32))

        self.play_icon = ImageTk.PhotoImage(self.play_icon)
        self.play_icon2 = ImageTk.PhotoImage(self.play_icon2)
        self.play_icon3 = ImageTk.PhotoImage(self.play_icon3)
        self.play_icon4 = ImageTk.PhotoImage(self.play_icon4)

        # ボタンの作成
        self.load_button = ttk.Button(parent, text="Load Video", command=self.load_video)
        self.load_button.pack(side=tk.TOP, pady=10)

        self.prev_frame_button = ttk.Button(parent, image=self.play_icon3, command=self.prev_frame)
        self.prev_frame_button.pack(side=tk.LEFT, padx=10)

        self.next_frame_button = ttk.Button(parent, image=self.play_icon, command=self.next_frame)
        self.next_frame_button.pack(side=tk.LEFT, padx=10)

        self.save_frame_button = ttk.Button(parent, text="Save Frame as JPEG", command=self.save_current_frame_as_jpeg)
        self.save_frame_button.pack(side=tk.LEFT, padx=10)

        # フレーム範囲を指定するGUI要素
        self.frame_range_label = ttk.Label(parent, text="Frame Range:")
        self.frame_range_label.pack(side=tk.LEFT, padx=10)

        self.start_frame_entry = ttk.Entry(parent)
        self.start_frame_entry.pack(side=tk.LEFT)

        self.to_label = ttk.Label(parent, text="to")
        self.to_label.pack(side=tk.LEFT)

        self.end_frame_entry = ttk.Entry(parent)
        self.end_frame_entry.pack(side=tk.LEFT)

        self.save_frame_range_button = ttk.Button(parent, text="Save Frames in Range", command=self.save_frames_in_range)
        self.save_frame_range_button.pack(side=tk.LEFT, padx=10)

        # スライダーの追加
        self.frame_slider = ttk.Scale(parent, from_=0, to=100, orient="horizontal", command=self.on_slider_move)
        self.frame_slider.pack(side=tk.BOTTOM, fill="x")

        # フレーム数を表示するラベルの追加
        self.frame_count_label = ttk.Label(parent, text="Total Frames: 0")
        self.frame_count_label.pack(side=tk.BOTTOM, padx=10)
        
           # フレーム数を表示するラベルを作成する
        self.total_frames_label = tk.Label(parent, text="Total Frames: 0")
        self.total_frames_label.pack()
        
         # ビデオを読み込んでフレーム数を取得し、ラベルに表示する
    
    def load_video(self):
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.play_video()
         
            capture = cv2.VideoCapture(self.video_path)   
             # 動画が再生されるときにフレーム数を更新する
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_frames_label.config(text=f"Total Frames: {total_frames}")


    
    def prev_frame(self):
        self.adjust_video_position(-1)
    
    def next_frame(self):
        self.adjust_video_position(1)

    def save_current_frame_as_jpeg(self):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                filename = filedialog.asksaveasfilename(defaultextension=".jpeg")
                if filename:
                    cv2.imwrite(filename, frame)

    def save_frames_in_range(self):
        if self.capture:
            selected_directory = filedialog.askdirectory()
            if not selected_directory:
                return
            start_frame = int(self.start_frame_entry.get())
            end_frame = int(self.end_frame_entry.get())
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = self.capture.read()
                if ret:
                    filename = os.path.join(selected_directory, f"frame_{current_frame}.jpeg")  # Create a unique file name for each frame
                    cv2.imwrite(filename, frame)  # Save the frame
                    current_frame += 1
                else:
                    break

    def on_slider_move(self, value):
        if self.capture:
            frame_pos = float(value)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_pos))
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.resize(frame, (self.video_width, self.video_height))  # Resize video frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(self.video_x, self.video_y, image=self.photo, anchor=tk.NW)

    def play_video(self):
        self.capture = cv2.VideoCapture(self.video_path)
        
        self.isPlaying = True
        self.play_frame()

        # 動画が再生されるときにフレーム数を更新する
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count_label.config(text=f"Total Frames: {total_frames}")

        # 追加機能：ファイル読み込み後の総フレーム数をラベルに表示する
        self.total_frames_label.config(text=f"Total Frames: {total_frames}")
    
    def play_frame(self):
        if self.isPlaying:
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.resize(frame, (self.video_width, self.video_height))  # Resize video frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(self.video_x, self.video_y, image=self.photo, anchor=tk.NW)
                self.frame_count += 1
                # フレーム数をラベルに表示
                self.frame_count_label.config(text=f"Total Frames: {self.frame_count}")
                self.parent.after(30, self.play_frame)

                # スライダーの最大値を動画のフレーム数に設定
                total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
                self.frame_slider.config(to=total_frames)

    def stop_video(self):
        if self.capture:
            self.isPlaying = False
            self.capture.release()
            self.canvas.delete("all")

    def play_pause_video(self):
        if self.isPlaying:
            self.isPlaying = False
            self.play_frame()
        else:
            self.isPlaying = True
            self.play_frame()
    
    def adjust_video_position(self, direction):
        if self.capture:
            current_frame_pos = self.capture.get(cv2.CAP_PROP_POS_FRAMES)
            new_frame_pos = max(0, int(current_frame_pos + direction - 1))
            print(new_frame_pos)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.resize(frame, (self.video_width, self.video_height))  # Resize video frame
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(self.video_x, self.video_y, image=self.photo, anchor=tk.NW)

   
class CustomThread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._run = self.run
        self.run = self.set_id_and_run

    def set_id_and_run(self):
        self.id = threading.get_native_id()
        self._run()

    def get_id(self):
        return self.id
        
    def raise_exception(self):
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_long(self.get_id()), 
            ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                ctypes.c_long(self.get_id()), 
                0
            )
            print('Failure in raising exception')

class VideoAnalyzer:
    def __init__(self, parent, row, column):
        self.parent = parent
        self.row = row
        self.column = column
        self.video_path = ""
        self.start_frame_number = 0

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
        
        self.video_thread = None
        self.stop_thread_flag = ""
        self.lock = threading.Lock()

    def get_input_text(self):
        return self.input_text.get()

    def select_video(self):
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread.raise_exception()
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.plot_audio_waveform()
            # スレッドがすでに実行中であれば停止
            # 新しいスレッドを開始
            self.stop_thread_flag = ""  # 新しいスレッド用にフラグをリセット
            self.video_thread = CustomThread(target=self.play_video)
            self.video_thread.start()

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
           
            self.sample_rate = sample_rate
            self.waveforms = waveforms
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
            start_frame_number = 0
            frame_size = (int)(self.sample_rate / 30)
            while True:
                flag = False
                for i in range(frame_size):
                    if  self.waveforms[start_frame_number + i] > 0.4:
                        flag = True
                        break
                if  flag == True:
                    break
                start_frame_number += frame_size
            self.start_frame_number = (int)(start_frame_number / frame_size)
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame_number)
            while self.stop_thread_flag != "stop":
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

def set_resolution():
    global resolution
    resolution_str = simpledialog.askstring("解像度を入力", "解像度を幅x高さの形式で入力してください（例：640x480）:")
    if resolution_str:
        width, height = map(int, resolution_str.split("x"))
        resolution = (width, height)


duration = 10
def set_duration():
    global duration
    duration_str = simpledialog.askstring("動画の長さを入力", "動画の長さ（秒）を入力してください:")
    if duration_str:
        duration = float(duration_str)


second_tab = None
def handle_plot_and_save():
    global second_tab
    files = filedialog.askopenfilenames()
    if files:
        for file_path in files:
            if file_path.endswith('.mp4'):
                probe = ffmpeg.probe(file_path, loglevel='error', select_streams='v:0',
                                     show_entries='frame=pkt_pos,best_effort_timestamp_time,pkt_duration_time')
                timestamps = [float(frame['best_effort_timestamp_time']) for frame in probe['frames']]
                timestamp_diffs = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                fig, ax = plt.subplots()
                ax.plot(timestamps[1:], timestamp_diffs, marker='o', linestyle='', markersize=1)
                ax.set_xlabel('Time stamps')
                ax.set_ylabel('Timestamp Difference')
                ax.set_title('Timestamp Differences Between Consecutive Frames')
                ax.grid(True)
                canvas = FigureCanvasTkAgg(fig, master=second_tab)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
                data = {"Timestamps": timestamps[1:], "Timestamp Differences": timestamp_diffs}
                df = pd.DataFrame(data)
                save_location = filedialog.asksaveasfilename(defaultextension=".xlsx")
                if save_location:
                    df.to_excel(save_location, index=False)

def handle_load():
    global table
    files = filedialog.askopenfilenames()
    for file in files:
        probe = ffmpeg.probe(file)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            duration = eval(video_stream['duration'])
            max_fps = eval(video_stream['r_frame_rate'])
            avg_fps = eval(video_stream['avg_frame_rate'])
            total_frame = duration * max_fps
            actual_frame = duration * avg_fps
            dropped_frame = 0 if actual_frame >= total_frame else total_frame - actual_frame
            drop_rate = dropped_frame / total_frame * 100
            filename = os.path.basename(file)
            table.insert("", "end", text=filename, values=(duration, '%.2f' % max_fps, '%.2f' % avg_fps,
                                                          int(dropped_frame), '%.2f' % drop_rate))

resolution = (640, 480)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

def merge_videos(video_analyzers):
    global duration
    global resolution
    video_count = 0
    for video_analyzer in video_analyzers:
        if video_analyzer.video_path != "":
            video_count += 1
    if video_count > 0:
        print("Merging videos...")
        caps = []
        labels = []
        for video_analyzer in video_analyzers:
            if video_analyzer.video_path == "":
                continue
            cap = cv2.VideoCapture(video_analyzer.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, video_analyzer.start_frame_number)  # 120フレーム目から読み込みを開始する
            caps.append(cap)
            labels.append(video_analyzer.get_input_text())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (FRAME_WIDTH * 2, FRAME_HEIGHT * 2))
        
        start_time = time.time()
        end_time = start_time + duration
        

        # 動画の再生と保存
        while time.time() < end_time:
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
                
            black_screen = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            if len(frames) == 1:
                output_frame = frames[0]
            elif len(frames) == 2:
                output_frame = np.hstack(frames)
            elif len(frames) == 3:
                frames.append(black_screen)
                output_frame = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:4])])
            elif len(frames) == 4:
                frame1, frame2, frame3, frame4 = frames
                output_frame = np.vstack([np.hstack([frame1, frame2]), np.hstack([frame3, frame4])])

            output_video.write(output_frame)
            
            
            resized_frame = cv2.resize(output_frame,resolution)
            cv2.imshow("Video", resized_frame)

            # ウィンドウを閉じるボタンが押された場合にウィンドウを閉じる
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # ウィンドウが閉じられた場合にループを終了する
            if cv2.getWindowProperty("Video", cv2.WND_PROP_VISIBLE) < 1:
                break
            time.sleep(1 / 33)

def analyze_audio_waveform(video_analyzer):
    start_frame_number = 0
    frame_size = (int)(video_analyzer.sample_rate / 30)
    while True:
        flag = False
        for i in range(frame_size):
            if  video_analyzer.waveforms[start_frame_number + i] > 0.4:
                flag = True
                break
        if  flag == True:
            break
        start_frame_number += frame_size
    start_frame_number = (int)(start_frame_number / frame_size)
    video_analyzer.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)
    while video_analyzer.stop_thread_flag != "stop":
        ret, frame = video_analyzer.capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (400, 300))
        video_analyzer.canvas.delete("all")
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        video_analyzer.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        video_analyzer.parent.update()
        time.sleep(1 / 30)

def main():
    global duration
    global second_tab
    window = tk.Tk()
    window.title("Video Helper (Ver0.2)")

    # ウィンドウのサイズと位置を設定
    window_width = 800
    window_height = 600
    window.geometry(f"{window_width}x{window_height}+{window_width}+{window_height}")

    tab_control = ttk.Notebook(window)
    tab_control.pack(expand=1, fill="both")

    first_tab = ttk.Frame(tab_control)
    first_tab.pack(expand=True, fill="both")
    tab_control.add(first_tab, text="Frame drop rate")

    load_button = ttk.Button(first_tab, text="Load", command=handle_load)
    load_button.pack()

    table = ttk.Treeview(first_tab)
    table["columns"] = ("Duration", "Max FPS", "Avg FPS", "Dropped Frames", "Drop Rate")
    table.heading("#0", text="File")
    table.heading("Duration", text="Duration")
    table.heading("Max FPS", text="Max FPS")
    table.heading("Avg FPS", text="Avg FPS")
    table.heading("Dropped Frames", text="Dropped Frames")
    table.heading("Drop Rate", text="Drop Rate (%)")
    table.pack()

    second_tab = ttk.Frame(tab_control)
    second_tab.pack(expand=True, fill="both")
    tab_control.add(second_tab, text="Analysis")

    plot_button = ttk.Button(second_tab, text="Load", command=handle_plot_and_save)
    plot_button.pack()

    third_tab = ttk.Frame(tab_control)
    third_tab.pack(expand=True, fill="both")
    tab_control.add(third_tab, text="Video Player")
    
    video_player = VideoPlayer(third_tab, width=800, height=500, video_width=700, video_height=400)
    
    fourth_tab = ttk.Frame(tab_control)
    tab_control.add(fourth_tab, text='Merge Videos')

    frame1 = tk.Frame(fourth_tab)
    frame1.pack()

    # VideoAnalyzerのインスタンスを作成して配置
    video_analyzers = []
    for i in range(2):
        for j in range(2):
            video_analyzer = VideoAnalyzer(frame1, j*2 + 1, i*2)
            video_analyzers.append(video_analyzer)

    # 解像度設定ボタンの作成と配置
    resolution_button = ttk.Button(fourth_tab, text="Set Resolution", command=set_resolution)
    resolution_button.pack()

    # 長さ設定ボタンの作成と配置
    duration_button = ttk.Button(fourth_tab, text="Set Duration", command=set_duration)
    duration_button.pack()

    # Merge Videoボタンの作成と配置
    merge_button = ttk.Button(fourth_tab, text="Merge Videos", command=lambda:merge_videos(video_analyzers))
    merge_button.pack()

    window.mainloop()

if __name__ == "__main__":
  main()
