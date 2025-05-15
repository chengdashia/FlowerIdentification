import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2lab


def remove_gray_background(image, threshold=40):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    R, G, B = cv2.split(image_rgb)
    max_rgb = np.maximum(np.maximum(R, G), B)
    min_rgb = np.minimum(np.minimum(R, G), B)
    diff = max_rgb - min_rgb
    mask = diff < threshold
    image_rgb[mask] = [255, 255, 255]
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)


def remove_green_region(image, lower_green, upper_green):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask_cleaned = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel, iterations=2)
    mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_DILATE, kernel, iterations=1)
    result = image.copy()
    result[mask_cleaned > 0] = [255, 255, 255]
    return result


def compute_lab(image_bgr, mask=None):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_lab = rgb2lab(image_rgb / 255.0)
    white_mask = np.all(image_rgb == [255, 255, 255], axis=-1)
    valid_mask = ~white_mask
    if mask is not None:
        valid_mask &= mask

    valid_L = image_lab[:, :, 0][valid_mask]
    valid_A = image_lab[:, :, 1][valid_mask]
    valid_B = image_lab[:, :, 2][valid_mask]

    if valid_L.size == 0:
        print("⚠️ 所选区域无有效像素（非白色）")
        return

    # ✅ 计算平均值
    mean_L = np.mean(valid_L)
    mean_A = np.mean(valid_A)
    mean_B = np.mean(valid_B)

    # ✅ 找到 a* 最大值对应的 LAB 值
    max_a_index = np.argmax(valid_A)
    max_L = valid_L[max_a_index]
    max_A = valid_A[max_a_index]
    max_B = valid_B[max_a_index]

    # ✅ 打印两个结果
    print(f"\n🌈 LAB 颜色均值：")
    print(f"▶ 平均 L*: {mean_L:.2f}")
    print(f"▶ 平均 a*: {mean_A:.2f}")
    print(f"▶ 平均 b*: {mean_B:.2f}")

    print(f"\n🔺 a* 最大值对应的 LAB 值：")
    print(f"▶ L*: {max_L:.2f}")
    print(f"▶ a*: {max_A:.2f}  ")
    print(f"▶ b*: {max_B:.2f}")


def main():
    image_path = r"./huasi/HS1.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print("❌ 无法读取图片，请检查路径")
        return

    image_no_gray = remove_gray_background(image, threshold=40)
    lower_green = np.array([22, 40, 40])
    upper_green = np.array([85, 255, 255])
    image_no_green = remove_green_region(image_no_gray, lower_green, upper_green)

    # 显示图像（不添加标题）
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(image_no_green, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

    # 直接计算整图 LAB 均值（去除白色区域）
    compute_lab(image_no_green)


if __name__ == "__main__":
    main()
