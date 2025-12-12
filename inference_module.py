import cv2
import numpy as np
import os
import re
import torch
import json
import base64
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import sys
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =========================================================
# 1. 환경 설정
# =========================================================

ORIGINAL_DIR = os.getcwd()
DTR_PATH = os.path.abspath('./deep-text-recognition-benchmark')
sys.path.insert(0, DTR_PATH)

os.chdir(DTR_PATH)
try:
    from inference import inference, load_text_recognition_model
    print("OCR 모듈 import 성공")
except ImportError as e:
    print(f"오류: {e}")
    print("   deep-text-recognition-benchmark 폴더 구조를 확인하세요.")
    print("   필요한 폴더: modules/, model.py, inference.py 등")
    sys.exit()
finally:
    os.chdir(ORIGINAL_DIR)  # 원래 디렉토리로 복귀

# [모델 경로]
YOLO_WEIGHTS = 'runs/detect/train6/weights/best.pt'
OCR_WEIGHTS = os.path.abspath("./weights/vgg__high_best_accuracy.pth")

# [파라미터]
CONFIDENCE_THRESHOLD = 0.35 
OCR_CONFIDENCE_THRESHOLD = 0.3
SKIP_FRAMES = 5

if os.name == 'nt': 
    FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
else: 
    FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OCR 전처리
ocr_transform = A.Compose([
    A.Resize(32, 100),
    A.Normalize(mean=0, std=1),
    ToTensorV2()
])

# 번호판 정규식 
plate_pattern = re.compile(r"\D{0,5}\d{0,3}\D{1}\d{4}$")

# =========================================================
# 함수 정의
# =========================================================

def image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def get_time_str(msec):
    seconds = int(msec / 1000)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def detect_with_slicing(model, frame, confidence=0.25):
    height, width, _ = frame.shape
    mid_x = width // 2  
    results_coords = [] 
    
    overlap = 100 
    left_img = frame[:, 0 : mid_x + overlap]
    results_left = model(left_img, conf=confidence, verbose=False)
    for result in results_left:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().tolist()
            conf = box.conf.item()
            results_coords.append(coords + [conf])

    right_img = frame[:, mid_x - overlap : width]
    results_right = model(right_img, conf=confidence, verbose=False)
    for result in results_right:
        for box in result.boxes:
            coords = box.xyxy[0].cpu().tolist()
            conf = box.conf.item()
            coords[0] += (mid_x - overlap)
            coords[2] += (mid_x - overlap)
            results_coords.append(coords + [conf])

    return results_coords

def transform_vertical_plate(plate_img):
    h, w, c = plate_img.shape
    aspect_ratio = w / h
    if aspect_ratio > 2.5: return plate_img

    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([15, 100, 100]), np.array([40, 255, 255]))
    
    if cv2.countNonZero(mask) / (w * h) > 0.3:
        split_point = int(w * 0.25) 
        left = plate_img[:, :split_point]
        right = plate_img[:, split_point:]
        top = cv2.resize(left[:h//2, :], (w//4, h))
        bot = cv2.resize(left[h//2:, :], (w//4, h))
        new_left = cv2.resize(np.hstack([top, bot]), (int(w*0.3), h))
        return np.hstack([new_left, right])
    return plate_img

def run_ocr(model, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    input_tensor = ocr_transform(image=gray)["image"]
    ocr_result, confidence_score = inference(model=model, input_tensor=input_tensor, device="cuda")
    
    text = ocr_result[0]
    conf = confidence_score[0]
    
    res = plate_pattern.match(text)
    if not (6 < len(text) < 11 and res):
        text = "invalid"
    
    return text, conf

def draw_text(img, text, x, y, color=(0,255,0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try: 
        font = ImageFont.truetype(FONT_PATH, 30)
    except: 
        font = ImageFont.load_default()
    draw.rectangle(draw.textbbox((x, y-35), text, font=font), fill=(0,0,0,150))
    draw.text((x, y-35), text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def enhance_for_motion_blur(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    gaussian = cv2.GaussianBlur(img, (0, 0), 3.0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    return img


# =========================================================
# 메인 추론 함수 (Streamlit에서 호출)
# =========================================================
def run_plate_detection(video_path, output_dir, progress_callback=None):
    """
    번호판 검출 추론 실행
    
    Args:
        video_path: 입력 영상 경로
        output_dir: 결과 저장 디렉토리
        progress_callback: 진행률 콜백 함수 (optional) - progress_callback(current, total)
    
    Returns:
        result_video_path: 결과 영상 경로
        result_json_path: 결과 JSON 경로
        json_data: 검출 결과 리스트
    """
    print(f"Device: {DEVICE}")
    print(f"입력 파일: {video_path}")
    
    # 출력 경로 설정
    file_name_with_ext = os.path.basename(video_path)
    file_name_only = os.path.splitext(file_name_with_ext)[0]
    
    os.makedirs(output_dir, exist_ok=True)
    capture_dir = os.path.join(output_dir, "captured_plates")
    os.makedirs(capture_dir, exist_ok=True)
    
    result_video_path = os.path.join(output_dir, f"{file_name_only}_result.mp4")
    result_json_path = os.path.join(output_dir, f"{file_name_only}_result.json")
    
    # 모델 로딩
    print("모델 로딩...")
    yolo = YOLO(YOLO_WEIGHTS)
    ocr_model = load_text_recognition_model(
        save_model=OCR_WEIGHTS,
        device="cuda"
    )
    print("OCR 모델 로드 완료")

    # 비디오 열기
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 임시 AVI 파일로 저장 (XVID 코덱 - 호환성 좋음)
    temp_video_path = result_video_path.replace('.mp4', '_temp.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print("XVID 실패, MJPG 시도...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        temp_video_path = result_video_path.replace('.mp4', '_temp.avi')
        out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    
    print(f"VideoWriter 열림: {out.isOpened()}")

    print("▶️ 실행 중...")
    frame_cnt = 0
    json_results = []
    rects_to_draw = [] 

    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        frame_cnt += 1
        
        current_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_str = get_time_str(current_ms)
        
        if frame_cnt % SKIP_FRAMES == 0:
            rects_to_draw = []
            detections = detect_with_slicing(yolo, frame, confidence=CONFIDENCE_THRESHOLD)

            for det in detections:
                x1, y1, x2, y2, det_conf = list(map(int, det[:4])) + [det[4]]
                
                w_box, h_box = x2-x1, y2-y1
                if w_box < 30 or h_box < 10: 
                    continue

                pad_w_left = int(w_box * 0.15)
                pad_w_right = int(w_box * 0.05)
                pad_h = int(h_box * 0.15)
                
                cx1 = max(0, x1 - pad_w_left)
                cy1 = max(0, y1 - pad_h)
                cx2 = min(width, x2 + pad_w_right)
                cy2 = min(height, y2 + pad_h)
                
                plate_roi = frame[cy1:cy2, cx1:cx2]
                if plate_roi.size == 0: 
                    continue

                if (cx2 - cx1) < 128:
                    plate_roi = cv2.resize(plate_roi, dsize=(0,0), fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)

                plate_roi = transform_vertical_plate(plate_roi)
                plate_roi = enhance_for_motion_blur(plate_roi)

                text, ocr_conf = run_ocr(ocr_model, plate_roi)

                if text != "invalid" and ocr_conf >= OCR_CONFIDENCE_THRESHOLD:
                    color = (0, 255, 0)
                    label = f"Plate: {text}"
                    print(f"👍 [{time_str}] {text} (Det: {det_conf:.2f}, OCR: {ocr_conf:.2f})")
                    
                    file_name = f"{text}_{frame_cnt}.jpg"
                    cv2.imwrite(os.path.join(capture_dir, file_name), plate_roi)
                    
                    record = {
                        "plate_number": text,
                        "timestamp": time_str,
                        "det_confidence": round(float(det_conf), 4),
                        "ocr_confidence": round(float(ocr_conf), 4),
                        "frame_index": frame_cnt,
                        "bbox": [cx1, cy1, cx2, cy2],
                        "image_base64": image_to_base64(plate_roi)
                    }
                    json_results.append(record)
                    
                    rects_to_draw.append({
                        'coords': (x1, y1, x2, y2),
                        'text': label,
                        'color': color
                    })

        # 시각화
        for item in rects_to_draw:
            rx1, ry1, rx2, ry2 = item['coords']
            col = item['color']
            txt = item['text']
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), col, 3)
            frame = draw_text(frame, txt, rx1, ry1, col)

        out.write(frame)
        
        # 진행률 콜백
        if progress_callback and frame_cnt % 1 == 0:
            progress_callback(frame_cnt, total_frames)
        
        if frame_cnt % 1 == 0: 
            print(f"Processing {frame_cnt}/{total_frames} ({time_str})...")

    cap.release()
    out.release()
    
    print(f"임시 영상 저장 완료: {temp_video_path}")
    
    # ffmpeg로 H.264 MP4 변환
    print("H.264 코덱으로 변환 중...")
    
    try:
        import subprocess
        import shutil
        
        # anaconda ffmpeg 경로 찾기
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is None:
            ffmpeg_path = 'ffmpeg'
        
        print(f"ffmpeg 경로: {ffmpeg_path}")
        
        cmd = [
            ffmpeg_path, '-y',
            '-i', temp_video_path,
            '-vcodec', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            result_video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and os.path.exists(result_video_path):
            os.remove(temp_video_path)  # 임시 파일 삭제
            print("H.264 변환 완료!")
        else:
            print(f"ffmpeg 변환 실패: {result.stderr}")
            # 실패 시 임시 파일을 결과로 사용
            if os.path.exists(temp_video_path):
                os.rename(temp_video_path, result_video_path)
    except Exception as e:
        print(f"변환 중 오류: {e}")
        if os.path.exists(temp_video_path):
            os.rename(temp_video_path, result_video_path)
    
    # JSON 저장
    print(f"JSON 파일 저장 중... (후처리 전: {len(json_results)}건)")
    
    # 후처리: 같은 프레임에서 동일한 번호판 중복 제거
    def remove_duplicates(results):
        """같은 프레임에서 동일한 번호판이 여러 번 검출된 경우, 신뢰도가 가장 높은 것만 유지"""
        from collections import defaultdict
        
        # (frame_index, plate_number) 기준으로 그룹화
        grouped = defaultdict(list)
        for item in results:
            key = (item["frame_index"], item["plate_number"])
            grouped[key].append(item)
        
        # 각 그룹에서 ocr_confidence가 가장 높은 것만 선택
        deduplicated = []
        for key, items in grouped.items():
            best = max(items, key=lambda x: x["ocr_confidence"])
            deduplicated.append(best)
        
        # frame_index 순으로 정렬
        deduplicated.sort(key=lambda x: x["frame_index"])
        return deduplicated
    
    json_results = remove_duplicates(json_results)
    print(f"후처리 완료: {len(json_results)}건")
    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, ensure_ascii=False, indent=4)
        
    print(f"모든 작업 완료!")
    print(f"   - 결과 영상: {result_video_path}")
    print(f"   - 결과 JSON: {result_json_path}")
    
    return result_video_path, result_json_path, json_results


# =========================================================
# 독립 실행 (command line 인자 지원)
# =========================================================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        # command line에서 실행: python inference_module.py <video_path> <output_dir>
        video_path = sys.argv[1]
        output_dir = sys.argv[2]
    else:
        # 기본값
        video_path = './주주주주행.mp4'
        output_dir = "results"
    
    run_plate_detection(video_path, output_dir)