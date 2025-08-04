import cv2
import matplotlib.pyplot as plt
import numpy as np
import imagingcontrol4 as ic4


class CameraControl:
    def __init__(self):
        self._init_connection()
        self._grabber_and_sink_set()
        
    
    def _init_connection(self):
        print('=====Camera connecting=====')
        if ic4.Library._core is None:
            ic4.Library.init()
            print('ic4 init finish.')
        else:
            print('ic4.Library has been initiated!')
            
        device_list = ic4.DeviceEnum.devices()
        
        if device_list is not None:
            self.device = device_list[0]
            print('devices 0 has been assigned to self.device')
        
        print('=====Camera connect finish=====')
        
    def _grabber_and_sink_set(self):
        # Open the selected device in a new Grabber
        if hasattr(self, 'grabber'):
            print('True')
        else:
            self.grabber = ic4.Grabber(self.device)       
            
        
        if self.grabber.is_device_open:
            print('grabber is opened')            
        
        else:
            self.grabber = ic4.Grabber(self.device)

        # Create a snap sink for manual buffer capture
        self.sink = ic4.SnapSink()
        
        self.grabber.stream_setup(self.sink)
    
    def get_image(self):
        buffer = self.sink.snap_single(1000)
        return buffer.numpy_copy()       
            
    
    def print_camera_info(self):
        print('=====Camera information=====')
        print(f'Model name:{self.device.model_name}')
        print(f'Serial:{self.device.serial}')
        print(f'Version: {self.device.version}')
        print('=====Camera info finish=====')
        
    def camera_exit(self):
        self.grabber.stream_stop()
        self.grabber.device_close()
        ic4.Library.exit()
        
            
cam = CameraControl()
cam.print_camera_info()

img = cam.get_image()
cam.camera_exit()


plt.imshow(img, cmap='gray')
plt.show()




def show_image(title, image):
    """
    自定義函數，用於顯示圖片。
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else 'viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()

def find_rectangles(img):
    """
    從指定圖片中找出矩形輪廓並標註。
    """
    try:
        # 1. 讀取圖片並備份一份彩色影像用於繪圖
        original_image = img
        if original_image is None:
            print(f"錯誤：無法讀取圖片")
            return
        
        # 將原始影像轉換為灰階

        
        # 2. 影像預處理：使用高斯模糊去除雜訊
        blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
        show_image("1. 灰階與高斯模糊後的影像", blurred_image)

        # 3. 邊緣檢測：使用 Canny 演算法
        # 閾值設定需要根據你的影像特性來調整
        canny_edges = cv2.Canny(blurred_image, 50, 150)
        show_image("2. Canny 邊緣檢測結果", canny_edges)
        
        # 4. 輪廓尋找
        # cv2.RETR_EXTERNAL 只尋找最外層的輪廓
        # cv2.CHAIN_APPROX_SIMPLE 壓縮輪廓的水平、垂直、對角線部分
        contours, _ = cv2.findContours(canny_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"找到 {len(contours)} 個輪廓。")
        
        # 創建一個空白影像用於繪製找到的輪廓
        contour_image = np.zeros_like(original_image)
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
        show_image("3. 原始輪廓", contour_image)

        # 5. 輪廓篩選與矩形判斷
        rectangular_contours = []
        for contour in contours:
            # 計算輪廓面積，篩選掉過小或過大的輪廓
            area = cv2.contourArea(contour)
            if area < 1000 or area > 100000:  # 這裡的面積閾值需要根據實際情況調整
                continue

            # 多邊形逼近：epsilon 參數決定逼近的精確度
            # 這裡使用周長的 4% 作為 epsilon，可以根據需要調整
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # 判斷逼近後的多邊形是否為四邊形（矩形）
            if len(approx) == 4:
                # 繪製出這個矩形，並計算長寬比來進一步判斷
                # x, y, w, h 是矩形的外接框
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                # 判斷長寬比是否在合理範圍內
                if 0.8 <= aspect_ratio <= 1.2:  # 這裡以長寬比接近 1 為例，可根據需求調整
                    rectangular_contours.append(approx)
        
        # 6. 標註最終結果
        result_image = original_image.copy()
        for rect_contour in rectangular_contours:
            # 繪製綠色輪廓
            cv2.drawContours(result_image, [rect_contour], -1, (0, 255, 0), 3)
            # 在中心點標註文字
            x, y, w, h = cv2.boundingRect(rect_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.putText(result_image, "Rectangle", (center_x - 50, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        print(f"最終找到 {len(rectangular_contours)} 個矩形。")
        show_image("4. 最終結果：找到並標註的矩形", cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    
    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    # 這裡請替換成你的圖片路徑
    # image_path = "your_test_image.jpg"
    find_rectangles(img)