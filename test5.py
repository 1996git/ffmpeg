import cv2
import ffmpeg
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import os
from test3 import VideoAnalyzer
from test3 import merge_videos

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

    def load_video(self):
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.play_video()

    def play_video(self):
        self.capture = cv2.VideoCapture(self.video_path)
        self.isPlaying = True
        self.play_frame()

        # 動画が再生されるときにフレーム数を更新する
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count_label.config(text=f"Total Frames: {total_frames}")

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

    def next_frame(self):
        self.adjust_video_position(1)

    def prev_frame(self):
        self.adjust_video_position(-1)

    def save_current_frame_as_jpeg(self):
        if self.capture:
            ret, frame = self.capture.read()
            if ret:
                filename = filedialog.asksaveasfilename(defaultextension=".jpeg")
                if filename:
                    cv2.imwrite(filename, frame)  # Convert BGR to RGB before saving

    def save_frames_in_range(self):
        if self.capture:
            start_frame = int(self.start_frame_entry.get())
            end_frame = int(self.end_frame_entry.get())
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            current_frame = start_frame
            while current_frame <= end_frame:
                ret, frame = self.capture.read()
                if ret:
                    filename = f"frame_{current_frame}.jpeg"  # Create a unique file name for each frame
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


def handle_plot_and_save():
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


window = tk.Tk()
window.title("Video Helper (Ver0.2)")

# 画面の幅と高さを取得
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()

# ウィンドウのサイズと位置を設定
window_width = 800
window_height = 600
x = (screen_width - window_width) // 2
y = (screen_height - window_height) // 2
window.geometry(f"{window_width}x{window_height}+{x}+{y}")

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
fourth_tab.pack(expand=True, fill="both")
tab_control.add(fourth_tab, text="Merge Videos")
video_analyzers = []
frame1 = tk.Frame(fourth_tab)
for i in range(2):
    for j in range(2):
        video_analyzer = VideoAnalyzer(frame1, i*2 + 1, j*2)
        video_analyzers.append(video_analyzer)
select_button = ttk.Button(frame1, text="Merge Video", command=lambda: merge_videos(video_analyzers))
select_button.grid(row=0, column=0, pady=0)
frame1.pack()

window.mainloop()
