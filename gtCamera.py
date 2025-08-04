import cv2
import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
from PIL import Image, ImageTk
import imagingcontrol4 as ic4
import time
import os
import numpy as np

class CameraControl:
    """
    一個使用 imagingcontrol4 控制工業相機的類別。
    """
    def __init__(self):
        """初始化相機函式庫並連接到第一個找到的設備。"""
        print('===== Camera Initializing =====')
        try:
            ic4.Library.init()
            device_list = ic4.DeviceEnum.devices()
            if not device_list:
                raise RuntimeError("找不到任何相機設備。")
            
            self.device = device_list[0]
            self.grabber = ic4.Grabber(self.device)
            print(f"成功連接到設備: {self.device.model_name} ({self.device.serial})")

            # *** 修正點 ***
            # 屬性是來自 device 物件，而非 grabber 物件
            self.properties = self.device.properties

        except Exception as e:
            print(f"相機初始化失敗: {e}")
            ic4.Library.exit()
            raise
        
        self.sink = None
        print('===== Camera Initialization Finished =====')

    def setup_stream(self):
        """設定相機串流並開始擷取。"""
        try:
            self.sink = ic4.SnapSink()
            self.grabber.stream_setup(self.sink)
            print("串流設定完成。")
        except Exception as e:
            print(f"串流設定失敗: {e}")
            self.release()
            raise

    def get_parameter(self, name: str):
        """
        讀取指定相機參數的值。

        Args:
            name (str): 參數的名稱 (例如 "ExposureTime", "Width")。

        Returns:
            回傳參數的目前值，如果參數不存在或不可讀則返回 None。
        """
        try:
            if self.properties.is_available(name) and self.properties.is_readable(name):
                return self.properties[name].value
            else:
                print(f"參數 '{name}' 不存在或不可讀取。")
                return None
        except ic4.IC4Exception as e:
            print(f"讀取參數 '{name}' 時發生錯誤: {e}")
            return None

    def set_parameter(self, name: str, value):
        """
        設定指定相機參數的值。

        Args:
            name (str): 參數的名稱 (例如 "ExposureTime")。
            value: 要設定的值。
        
        Returns:
            bool: 設定成功返回 True，失敗返回 False。
        """
        try:
            if self.properties.is_available(name) and self.properties.is_writable(name):
                prop = self.properties[name]
                prop_type = prop.type

                # 對於枚舉類型(String)，直接賦值字串
                if prop_type == ic4.PropType.STRING and name in ["Width", "Height"]:
                     # 解析度通常是整數，但此處作為範例保留
                     pass

                elif prop_type == ic4.PropType.INTEGER:
                    value = int(float(value)) # 先轉 float 再轉 int 以支援科學記號
                elif prop_type == ic4.PropType.FLOAT:
                    value = float(value)
                
                min_val, max_val = prop.min, prop.max
                if min_val is not None and value < min_val:
                    print(f"設定失敗：值 {value} 小於允許的最小值 {min_val}。")
                    return False
                if max_val is not None and value > max_val:
                    print(f"設定失敗：值 {value} 大於允許的最大值 {max_val}。")
                    return False

                prop.value = value
                print(f"成功設定參數 '{name}' 為 {value}。")
                return True
            else:
                print(f"參數 '{name}' 不存在或不可寫入。")
                return False
        except (ic4.IC4Exception, ValueError) as e:
            print(f"設定參數 '{name}' 為 '{value}' 時發生錯誤: {e}")
            return False

    def list_available_parameters(self) -> list:
        """返回所有可用參數的列表。"""
        return [p.name for p in self.properties]

    def get_image(self, timeout_ms=2000) -> np.ndarray | None:
        """擷取單張影像。"""
        try:
            if not self.grabber.is_streaming:
                self.grabber.stream_start()
            buffer = self.sink.snap_single(timeout_ms)
            return buffer.numpy_copy()
        except ic4.IC4Exception as e:
            print(f"擷取影像時發生錯誤: {e}")
            return None

    def start_streaming(self):
        """開始相機串流。"""
        if not self.grabber.is_streaming:
            self.grabber.stream_start()
            print("相機串流已啟動。")

    def stop_streaming(self):
        """停止相機串流。"""
        if self.grabber.is_streaming:
            self.grabber.stream_stop()
            print("相機串流已停止。")

    def release(self):
        """停止串流、關閉設備並釋放函式庫資源。"""
        print('===== Releasing Camera Resources =====')
        if hasattr(self, 'grabber') and self.grabber.is_streaming:
            self.grabber.stream_stop()
        if hasattr(self, 'grabber') and self.grabber.is_device_open:
            self.grabber.device_close()
        ic4.Library.exit()
        print('===== Camera Resources Released =====')

class CameraApp:
    """
    一個整合 CameraControl 的 Tkinter GUI 應用程式。
    """
    def __init__(self, root):
        self.root = root
        self.root.title("相機即時影像與參數控制")
        self.cam = None
        self.is_streaming = False
        
        main_frame = tk.Frame(self.root)
        main_frame.pack(padx=10, pady=10)
        
        video_frame = tk.Frame(main_frame)
        video_frame.pack(side=tk.LEFT, padx=10)
        self.video_label = tk.Label(video_frame)
        self.video_label.pack()

        control_panel = tk.Frame(main_frame)
        control_panel.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        file_frame = tk.LabelFrame(control_panel, text="檔案控制")
        file_frame.pack(fill=tk.X, pady=5)
        self.snap_button = tk.Button(file_frame, text="擷取並存檔", command=self.snap_and_save)
        self.snap_button.pack(pady=5, padx=5, fill=tk.X)
        self.exit_button = tk.Button(file_frame, text="離開", command=self.on_closing)
        self.exit_button.pack(pady=5, padx=5, fill=tk.X)

        param_frame = tk.LabelFrame(control_panel, text="參數控制")
        param_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(param_frame, text="參數名稱:").grid(row=0, column=0, padx=5, pady=2, sticky='w')
        self.param_name_entry = tk.Entry(param_frame)
        self.param_name_entry.grid(row=0, column=1, padx=5, pady=2)
        self.param_name_entry.insert(0, "ExposureTime")

        tk.Label(param_frame, text="參數值:").grid(row=1, column=0, padx=5, pady=2, sticky='w')
        self.param_value_entry = tk.Entry(param_frame)
        self.param_value_entry.grid(row=1, column=1, padx=5, pady=2)

        tk.Label(param_frame, text="讀取結果:").grid(row=2, column=0, padx=5, pady=2, sticky='w')
        self.param_result_label = tk.Label(param_frame, text="N/A", fg="blue")
        self.param_result_label.grid(row=2, column=1, padx=5, pady=2, sticky='w')

        read_button = tk.Button(param_frame, text="讀取", command=self._read_parameter_from_gui)
        read_button.grid(row=3, column=0, pady=5, padx=5)
        set_button = tk.Button(param_frame, text="設定", command=self._write_parameter_from_gui)
        set_button.grid(row=3, column=1, pady=5, padx=5)
        
        list_button = tk.Button(param_frame, text="列出所有可用參數", command=self._list_all_params)
        list_button.grid(row=4, columnspan=2, pady=5, padx=5, sticky='ew')
        
        try:
            self.cam = CameraControl()
            self.cam.setup_stream()
            self.is_streaming = True
            self.cam.start_streaming()
            self.update_video()
        except Exception as e:
            messagebox.showerror("相機錯誤", f"無法初始化相機: {e}")
            self.root.destroy()
            return
            
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _read_parameter_from_gui(self):
        param_name = self.param_name_entry.get()
        if not param_name:
            messagebox.showwarning("輸入錯誤", "請輸入參數名稱。")
            return
        value = self.cam.get_parameter(param_name)
        if value is not None:
            self.param_result_label.config(text=str(value))
            self.param_value_entry.delete(0, tk.END)
            self.param_value_entry.insert(0, str(value))
            messagebox.showinfo("讀取成功", f"參數 '{param_name}' 的值為: {value}")
        else:
            self.param_result_label.config(text="讀取失敗")
            messagebox.showerror("讀取失敗", f"無法讀取參數 '{param_name}'。")

    def _write_parameter_from_gui(self):
        param_name = self.param_name_entry.get()
        param_value_str = self.param_value_entry.get()
        if not param_name or not param_value_str:
            messagebox.showwarning("輸入錯誤", "請輸入參數名稱和值。")
            return
        
        success = self.cam.set_parameter(param_name, param_value_str)
        if success:
            messagebox.showinfo("設定成功", f"成功設定參數 '{param_name}'。")
            new_value = self.cam.get_parameter(param_name)
            if new_value is not None:
                self.param_result_label.config(text=str(new_value))
        else:
            messagebox.showerror("設定失敗", f"無法設定參數 '{param_name}'。\n請檢查主控台輸出以獲取詳細資訊。")

    def _list_all_params(self):
        params = self.cam.list_available_parameters()
        list_win = tk.Toplevel(self.root)
        list_win.title("所有可用參數")
        list_win.geometry("400x600")
        txt = scrolledtext.ScrolledText(list_win, width=50, height=40)
        txt.pack(expand=True, fill='both')
        param_text = "\n".join(sorted(params))
        txt.insert(tk.INSERT, param_text)
        txt.config(state=tk.DISABLED)

    def update_video(self):
        if not self.is_streaming: return
        frame = self.cam.get_image(timeout_ms=500)
        if frame is not None:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            w, h = img_pil.size
            scale = 480 / h
            new_w, new_h = int(w * scale), int(h * scale)
            img_pil_resized = img_pil.resize((new_w, new_h), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img_pil_resized)
            self.video_label.config(image=img_tk)
            self.video_label.image = img_tk
        self.root.after(30, self.update_video)

    def snap_and_save(self):
        print("正在擷取單張影像...")
        frame = self.cam.get_image(timeout_ms=2000)
        if frame is not None:
            if not os.path.exists("img"): os.makedirs("img")
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join("img", f"capture_{timestamp}.jpg")
            try:
                cv2.imwrite(filename, frame)
                messagebox.showinfo("成功", f"影像已儲存至:\n{filename}")
                print(f"影像成功儲存至 {filename}")
            except Exception as e:
                messagebox.showerror("存檔失敗", f"無法儲存影像: {e}")
        else:
            messagebox.showwarning("擷取失敗", "無法從相機獲取影像。")

    def on_closing(self):
        if messagebox.askokcancel("離開", "您確定要離開嗎？"):
            self.is_streaming = False
            if self.cam:
                self.cam.release()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()
