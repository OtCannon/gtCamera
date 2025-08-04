import cv2
import numpy as np
import os
from mura_inspector import MuraInspector # 從 Canvas 的檔案中匯入類別

class FeatureMatcher:
    """
    一個使用 ORB 特徵來匹配物件的類別。

    這個類別會接收一個模板影像 (template image)，然後可以在一張更大的
    目標影像 (target image) 中找到該模板的位置。
    ORB 是 SURF 的一個高效、免費的替代方案。
    """

    def __init__(self, template_image: np.ndarray, n_features: int = 2000):
        """
        初始化 FeatureMatcher。

        Args:
            template_image (np.ndarray): 要尋找的模板影像 (BGR 格式)。
            n_features (int): ORB 演算法要偵測的最大特徵點數量。
        """
        if template_image is None:
            raise ValueError("模板影像不可為 None。")

        self.template_image_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
        self.template_h, self.template_w = self.template_image_gray.shape
        
        # 1. 建立 ORB 物件
        # ORB 是標準 OpenCV 的一部分，不需安裝 contrib 版本
        try:
            self.detector = cv2.ORB_create(nfeatures=n_features)
        except cv2.error as e:
            print(f"錯誤：無法建立 ORB 物件: {e}")
            raise

        # 2. 尋找模板的特徵點與描述子 (descriptors)
        self.kp_template, self.des_template = self.detector.detectAndCompute(self.template_image_gray, None)
        
        if self.des_template is None:
             raise ValueError(f"在模板影像中找不到任何特徵點，無法進行初始化。")

        print(f"在模板影像中找到 {len(self.kp_template)} 個 ORB 特徵點。")

    def find_object_in_image(self, target_image: np.ndarray, min_match_count: int = 10) -> np.ndarray | None:
        """
        在目標影像中尋找模板物件。

        Args:
            target_image (np.ndarray): 待檢測的目標影像 (BGR 格式)。
            min_match_count (int): 認定找到物件所需的最小匹配點數量。

        Returns:
            np.ndarray | None: 如果找到物件，返回繪製了邊界框和匹配線的目標影像；否則返回 None。
        """
        if target_image is None:
            print("錯誤：目標影像為 None。")
            return None
        
        print("\n開始在目標影像中尋找物件...")
        target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        # 1. 尋找目標影像的特徵點與描述子
        kp_target, des_target = self.detector.detectAndCompute(target_image_gray, None)
        print(f"在目標影像中找到 {len(kp_target)} 個 ORB 特徵點。")

        if des_target is None or len(des_target) < 2:
            print("在目標影像中找到的特徵點不足，無法進行匹配。")
            return None

        # 2. 使用 BFMatcher (Brute-Force Matcher) 進行特徵匹配
        # 對於 ORB 這類二元描述子，推薦使用 NORM_HAMMING 距離
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(self.des_template, des_target, k=2)

        # 3. 根據 Lowe's ratio test 篩選好的匹配點
        good_matches = []
        if matches and len(matches[0]) == 2:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        print(f"找到 {len(good_matches)} 個符合條件的匹配點。")

        # 4. 如果好的匹配點數量足夠，則計算單應性矩陣 (Homography)
        if len(good_matches) >= min_match_count:
            # 取得匹配點在兩張影像中的座標
            src_pts = np.float32([self.kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_target[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 計算單應性矩陣
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            if M is None:
                print("計算單應性矩陣失敗。")
                return None

            # 繪製結果
            matches_mask = mask.ravel().tolist()

            # 取得模板影像的四個角點
            h, w = self.template_image_gray.shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            
            # 使用單應性矩陣變換角點，找到在目標影像中的對應位置
            dst = cv2.perspectiveTransform(pts, M)

            # 在目標影像上繪製多邊形邊界
            result_image = cv2.polylines(target_image, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # 繪製匹配的特徵點連線
            draw_params = dict(matchColor=(0, 255, 0),  # 匹配線的顏色
                               singlePointColor=None,
                               matchesMask=matches_mask,  # 只畫出內點 (inliers)
                               flags=2)
            
            # 將模板影像和繪製了邊界的目標影像並排顯示
            template_bgr = cv2.cvtColor(self.template_image_gray, cv2.COLOR_GRAY2BGR)
            match_visualization = cv2.drawMatches(template_bgr, self.kp_template, result_image, kp_target, good_matches, None, **draw_params)

            return match_visualization

        else:
            print(f"找到的匹配點不足 {min_match_count} 個，無法可靠地定位物件。")
            return None

# --------------------------------------------------------------------------------
# 主程式：範例使用方式
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    # --- 步驟 1: 使用 MuraInspector 產生模板影像 ---
    
    TEMPLATE_SOURCE_PATH = "template_calibrated.jpg" # 替換成你的標準品影像路徑
    
    if not os.path.exists(TEMPLATE_SOURCE_PATH):
        print(f"錯誤：找不到模板來源影像 '{TEMPLATE_SOURCE_PATH}'")
    else:
        print("--- 步驟 1: 正在從來源影像中提取模板 ---")
        # 建立 MuraInspector 物件
        inspector = MuraInspector(image_path=TEMPLATE_SOURCE_PATH)
        # 執行處理流程，但不顯示中間步驟以加速
        template = inspector.process_image(show_steps=False)

        if template is not None:
            print("模板提取成功！")
            cv2.imshow("Extracted Template", template)
            cv2.waitKey(1) # 短暫顯示一下

            # --- 步驟 2: 使用 FeatureMatcher 在新影像中尋找模板 ---
            TARGET_IMAGE_PATH = "test_image.jpg" # 替換成你的待測影像路徑
            
            if not os.path.exists(TARGET_IMAGE_PATH):
                 print(f"錯誤：找不到目標影像 '{TARGET_IMAGE_PATH}'")
            else:
                target_img = cv2.imread(TARGET_IMAGE_PATH)

                # 建立 FeatureMatcher 物件
                matcher = FeatureMatcher(template_image=template)

                # 執行尋找
                result_visualization = matcher.find_object_in_image(target_img)

                # --- 步驟 3: 顯示最終匹配結果 ---
                if result_visualization is not None:
                    print("\n物件尋找成功！顯示匹配結果。")
                    # 調整視窗大小以適應螢幕
                    h, w = result_visualization.shape[:2]
                    scale = 960 / w if w > 0 else 1 # 將寬度縮放到 960 像素
                    small_result = cv2.resize(result_visualization, (int(w*scale), int(h*scale)))
                    
                    cv2.imshow("ORB Feature Matching Result", small_result)
                    cv2.waitKey(0)
                else:
                    print("\n在目標影像中未找到物件。")
        else:
            print("模板提取失敗，無法繼續執行特徵匹配。")

    cv2.destroyAllWindows()
