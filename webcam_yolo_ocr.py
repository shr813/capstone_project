import cv2
import numpy as np
import pytesseract


def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

    # 밝기 조절
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 샤프닝 필터
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)

    # adaptive threshold
    threshed = cv2.adaptiveThreshold(sharpened, 255,
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
    return threshed


# 이미지 경로
image_path = "image.jpg"
image = cv2.imread(image_path)
if image is None:
    print("❌ 이미지 로드 실패. 경로 확인")
    exit()

# 전처리
deskewed = deskew_image(image)
processed = preprocess(deskewed)

# OCR 실행
custom_config = r'--oem 3 --psm 11'  # sparse text
data = pytesseract.image_to_data(processed, lang='kor+eng', config=custom_config, output_type=pytesseract.Output.DICT)

# 결과 출력
output_text = ""
for i, word in enumerate(data['text']):
    if word.strip():
        output_text += word + " "
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, word, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

print("🧾 추출된 텍스트:\n")
print(output_text.strip())

# 결과 화면 표시
cv2.imshow("OCR Improved", cv2.resize(image, (960, 720)))
cv2.waitKey(0)
cv2.destroyAllWindows()
