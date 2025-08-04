import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class ROI_selector:
    """
    一個用於顯示器 Mura 檢測的影像處理類別。

    這個類別封裝了從讀取影像、自動尋找目標區域、進行第一次透視變換校正，
    到提供手動介面讓使用者精細選點、確認選擇，並進行第二次透視變換以獲得
    最終結果並存檔的完整流程。
    """

    def __init__(self, image_path: str):
        """
        初始化 MuraInspector。

        Args:
            image_path (str): 要處理的目標影像檔案路徑。
        """
        # --- 核心屬性 ---
        self.image_path = image_path
        self.original_image = None
        self.original_color_image = None
        self.transformed_image_first_pass = None
        self.final_transformed_image = None

        # --- Tkinter 介面相關屬性 ---
        self.tk_root = None
        self.tk_canvas = None
        self.tk_image_tk = None
        self.selected_points = []
        self.selection_done = False

    # --------------------------------------------------------------------------------
    # Tkinter 介面相關方法 (私有)
    # --------------------------------------------------------------------------------

    def _on_image_click(self, event):
        """
        Tkinter 畫布上的滑鼠點擊事件處理函數。
        """
        if len(self.selected_points) < 4:
            x, y = event.x, event.y
            self.selected_points.append((x, y))
            print(f"點擊座標: ({x}, {y}) - 已選擇 {len(self.selected_points)} 個點")

            # 在畫布上繪製點
            self.tk_canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill="red", outline="red", width=2)
            
            if len(self.selected_points) == 4:
                print("已選擇所有四個點。請點擊 '完成選點' 按鈕繼續。")

    def _on_done_button_click(self):
        """
        「完成選點」按鈕的點擊事件處理函數。
        """
        if len(self.selected_points) < 4:
            messagebox.showwarning("提示", "請先點擊影像上的四個角點。")
            return
        self.selection_done = True
        self.tk_root.destroy()

    def _start_manual_selection(self, image_to_display: np.ndarray) -> list | None:
        """
        啟動 Tkinter 介面讓使用者手動選擇四個點。

        Args:
            image_to_display (np.ndarray): 要顯示在 Tkinter 視窗中的影像 (BGR 格式)。

        Returns:
            list | None: 如果成功選擇四個點，返回點的列表；否則返回 None。
        """
        self.selected_points.clear()
        self.selection_done = False

        self.tk_root = tk.Tk()
        self.tk_root.title("請依序點擊四個角點 (左上, 右上, 右下, 左下)")

        img_pil = Image.fromarray(cv2.cvtColor(image_to_display, cv2.COLOR_BGR2RGB))
        self.tk_image_tk = ImageTk.PhotoImage(image=img_pil)

        self.tk_canvas = tk.Canvas(self.tk_root, width=img_pil.width, height=img_pil.height)
        self.tk_canvas.pack()
        self.tk_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image_tk)
        self.tk_canvas.bind("<Button-1>", self._on_image_click)

        done_button = tk.Button(self.tk_root, text="完成選點", command=self._on_done_button_click)
        done_button.pack(pady=10)

        # 手動管理 Tkinter 事件迴圈，直到選點完成或視窗關閉
        while not self.selection_done:
            try:
                self.tk_root.update_idletasks()
                self.tk_root.update()
            except tk.TclError:
                print("Tkinter 視窗已由使用者關閉。")
                self.selection_done = False
                break
        
        if len(self.selected_points) == 4 and self.selection_done:
            return self.selected_points
        else:
            print("手動選點未完成或被取消。")
            return None

    # --------------------------------------------------------------------------------
    # 影像處理核心方法 (私有)
    # --------------------------------------------------------------------------------

    def _show_image(self, title: str, image: np.ndarray, is_gray: bool = False):
        """
        使用 Matplotlib 顯示影像的輔助函數。
        """
        plt.figure(figsize=(8, 8))
        if is_gray:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title, fontsize=16)
        plt.axis('off')
        plt.show()
        plt.close()

    def _load_and_preprocess(self) -> bool:
        """
        讀取並預處理影像。
        """
        try:
            self.original_image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if self.original_image is None:
                raise FileNotFoundError(f"錯誤：無法讀取灰階圖片，請檢查路徑：{self.image_path}")
            
            self.original_color_image = cv2.imread(self.image_path)
            if self.original_color_image is None:
                raise FileNotFoundError(f"錯誤：無法讀取彩色圖片，請檢查路徑：{self.image_path}")
            
            return True
        except Exception as e:
            print(e)
            return False

    def _find_initial_rectangle(self, show_steps: bool) -> np.ndarray | None:
        """
        自動尋找影像中最大、最像矩形的輪廓。
        """
        blurred_image = cv2.GaussianBlur(self.original_image, (5, 5), 0)
        thresholded_image = cv2.adaptiveThreshold(
            blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, -5
        )
        kernel = np.ones((5, 5), np.uint8)
        closed_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return None
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in sorted_contours[:3]:
            if cv2.contourArea(contour) < 20000: continue
            epsilon = 0.06 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                rect = cv2.minAreaRect(approx)
                size = rect[1]; width, height = sorted(size)
                if width == 0 or height == 0: continue
                area = cv2.contourArea(contour); min_rect_area = width * height
                area_ratio = area / min_rect_area; aspect_ratio = height / width
                if 0.85 < area_ratio < 1.15 and 0.8 < aspect_ratio < 1.25:
                    return np.int32(cv2.boxPoints(rect))
        return None

    def _perform_perspective_transform(self, image_src: np.ndarray, points: np.ndarray, output_size: int) -> np.ndarray:
        """
        執行透視變換的通用函數。
        """
        pts1 = np.float32(points)
        ordered_pts = np.zeros((4, 2), dtype="float32")
        s = pts1.sum(axis=1)
        ordered_pts[0] = pts1[np.argmin(s)]; ordered_pts[2] = pts1[np.argmax(s)]
        diff = np.diff(pts1, axis=1)
        ordered_pts[1] = pts1[np.argmin(diff)]; ordered_pts[3] = pts1[np.argmax(diff)]
        pts2 = np.float32([[0, 0], [output_size - 1, 0], [output_size - 1, output_size - 1], [0, output_size - 1]])
        matrix = cv2.getPerspectiveTransform(ordered_pts, pts2)
        return cv2.warpPerspective(image_src, matrix, (output_size, output_size))

    # --------------------------------------------------------------------------------
    # 公開方法
    # --------------------------------------------------------------------------------

    def process_image(self, show_steps: bool = False, output_filename: str = "./img/final_result.jpg") -> np.ndarray | None:
        """
        執行完整的影像處理流程。

        Args:
            show_steps (bool): 是否顯示每一步處理過程的影像。預設為 False。
            output_filename (str): 最終結果的存檔名稱。預設為 "final_result.jpg"。

        Returns:
            np.ndarray | None: 成功時返回最終處理好的影像，失敗或取消則返回 None。
        """
        print("--- 開始 Mura 檢測流程 ---")
        if not self._load_and_preprocess():
            return None

        # --- 第一階段：自動檢測與第一次變換 ---
        initial_corners = self._find_initial_rectangle(show_steps=False) # 通常第一步不需要展示
        if initial_corners is None:
            print("自動檢測失敗，將直接在原始影像上啟動手動選點模式。")
            base_image_for_manual_selection = self.original_color_image
        else:
            print("自動檢測成功！執行第一次透視變換...")
            self.transformed_image_first_pass = self._perform_perspective_transform(
                self.original_color_image, initial_corners, 400
            )
            base_image_for_manual_selection = self.transformed_image_first_pass

        # --- 第二階段：手動選點、確認與第二次變換 ---
        print("\n--- 階段 2: 手動精細選點 ---")
        manual_points = self._start_manual_selection(base_image_for_manual_selection)

        if manual_points is None:
            print("流程終止。")
            return None
        
        # <<< 新增功能：確認對話框 >>>
        # 建立一個暫時的 Tkinter root 來顯示 messagebox，然後立即銷毀
        root = tk.Tk()
        root.withdraw() # 隱藏主視窗
        confirm = messagebox.askyesno("確認選點", "您確定要使用這四個點進行最終校正嗎？", parent=root)
        root.destroy()

        if not confirm:
            print("使用者取消了選擇，流程終止。")
            return None
        # <<< 功能結束 >>>

        print("手動選點已確認！開始執行第二次精細透視變換...")
        
        self.final_transformed_image = self._perform_perspective_transform(
            base_image_for_manual_selection, np.array(manual_points), 300
        )
        
        # <<< 新增功能：儲存檔案 >>>
        try:
            cv2.imwrite(output_filename, self.final_transformed_image)
            print(f"成功！最終結果已儲存為 '{output_filename}'")
        except Exception as e:
            print(f"錯誤：儲存檔案 '{output_filename}' 失敗: {e}")
        # <<< 功能結束 >>>

        print("\n--- Mura 檢測流程完成！ ---")
        if show_steps:
            self._show_image("最終精細裁剪結果", self.final_transformed_image)

        return self.final_transformed_image

# --------------------------------------------------------------------------------
# 主程式：範例使用方式
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    IMAGE_PATH = "./img/capture_20250805_021931.jpg" # 請替換成你的圖片路徑

    inspector = ROI_selector(image_path=IMAGE_PATH)

    # 執行處理流程，並顯示中間步驟
    final_result = inspector.process_image(show_steps=True, output_filename="./img/image.jpg")

    if final_result is not None:
        print("\n處理成功！")
        # 即使存檔了，也用視窗顯示一下最終結果
        cv2.imshow("Final Calibrated Image", final_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n處理失敗或被使用者取消。")
