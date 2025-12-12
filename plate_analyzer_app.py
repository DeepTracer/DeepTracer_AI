import streamlit as st
import json
import base64
import os
import tempfile
import cv2
from datetime import timedelta
from PIL import Image
import io
import subprocess
import time

# =========================================================
# 페이지 설정
# =========================================================
st.set_page_config(
    page_title="번호판 분석 서비스",
    page_icon="icon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 커스텀 CSS
# =========================================================
st.markdown("""
<style>
    /* 전체 배경 */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* 사이드바 스타일 */
    [data-testid="stSidebar"] {
        background-color: #121212;
        border-right: 1px solid #2a2a2a;
    }
    
    /* 카드 스타일 */
    .card {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #2a2a2a;
        margin-bottom: 16px;
    }
    
    .card-title {
        color: #ffffff;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    /* 번호판 카드 */
    .plate-card {
        background-color: #1a1a1a;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2a2a2a;
    }
    
    .plate-number {
        color: #ffffff;
        font-size: 28px;
        font-weight: 700;
        font-family: monospace;
        letter-spacing: 2px;
    }
    
    .plate-label {
        color: #888888;
        font-size: 13px;
    }
    
    .confidence-high { color: #22c55e; }
    .confidence-mid { color: #eab308; }
    .confidence-low { color: #ef4444; }
    
    /* 버튼 스타일 */
    .stButton > button {
        background-color: #2a2a2a;
        border: 1px solid #3a3a3a;
        color: #e0e0e0;
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #3a3a3a;
        border-color: #4a4a4a;
    }
    
    /* 숨기기 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* 번호판 인덱스 표시 */
    .plate-index {
        background-color: #ff6b35;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    
    /* 네비게이션 버튼 */
    .nav-btn {
        background-color: #2a2a2a !important;
        border: 1px solid #3a3a3a !important;
        padding: 8px 20px !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# 세션 상태 초기화
# =========================================================
if "video_path" not in st.session_state:
    st.session_state["video_path"] = None
if "video_bytes" not in st.session_state:
    st.session_state["video_bytes"] = None
if "result_video_path" not in st.session_state:
    st.session_state["result_video_path"] = None
if "json_data" not in st.session_state:
    st.session_state["json_data"] = None
if "json_path" not in st.session_state:
    st.session_state["json_path"] = None
if "current_frame" not in st.session_state:
    st.session_state["current_frame"] = 0
if "total_frames" not in st.session_state:
    st.session_state["total_frames"] = 100
if "fps" not in st.session_state:
    st.session_state["fps"] = 30
if "current_plate_idx" not in st.session_state:
    st.session_state["current_plate_idx"] = 0
if "processing_done" not in st.session_state:
    st.session_state["processing_done"] = False


# =========================================================
# 유틸리티 함수
# =========================================================
def decode_base64_image(base64_str):
    """Base64 문자열을 PIL Image로 변환"""
    try:
        img_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(img_data))
    except Exception:
        return None


def get_video_info(video_path):
    """비디오 정보 추출"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()
    return fps, total_frames, duration


def get_frame_at_index(video_path, frame_idx):
    """특정 프레임 추출"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def find_plates_for_frame(json_data, frame_idx, frame_range=5):
    """현재 프레임 근처의 번호판 찾기"""
    if not json_data:
        return []
    result = []
    for p in json_data:
        p_frame = p.get("frame_index", p.get("frame", 0))
        if abs(p_frame - frame_idx) <= frame_range:
            result.append(p)
    return result


def format_time(seconds):
    """시간 포맷팅"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_confidence_color(conf):
    """신뢰도에 따른 색상 클래스"""
    if conf >= 0.8:
        return "confidence-high"
    elif conf >= 0.5:
        return "confidence-mid"
    return "confidence-low"


def get_confidence_color_hex(conf):
    """신뢰도에 따른 색상 HEX"""
    if conf >= 0.8:
        return "#22c55e"
    elif conf >= 0.5:
        return "#eab308"
    return "#ef4444"


# =========================================================
# 추론 실행 함수
# =========================================================
def run_inference(video_path, output_dir, progress_bar=None, progress_text=None):
    """
    추론 실행 - subprocess로 inference_module.py 호출
    """
    import subprocess
    
    try:
        # 결과 파일 경로 미리 계산
        file_name = os.path.splitext(os.path.basename(video_path))[0]
        result_video = os.path.join(output_dir, f"{file_name}_result.mp4")
        result_json = os.path.join(output_dir, f"{file_name}_result.json")
        
        # 총 프레임 수 계산
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # subprocess로 추론 모듈 실행
        current_dir = os.path.dirname(os.path.abspath(__file__))
        inference_script = os.path.join(current_dir, "inference_module.py")
        
        process = subprocess.Popen(
            ["python", inference_script, video_path, output_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 실시간으로 출력 읽으면서 진행률 업데이트
        for line in process.stdout:
            print(line.strip())
            
            # "Processing 현재/전체" 패턴 찾기
            if "Processing" in line and "/" in line:
                try:
                    parts = line.split("Processing")[1].strip().split("/")
                    current_frame = int(parts[0].strip())
                    
                    progress = current_frame / total_frames
                    if progress_bar:
                        progress_bar.progress(min(progress, 0.95))
                    if progress_text:
                        progress_text.text(f"{current_frame}/{total_frames} ({progress*100:.1f}%)")
                except:
                    pass
        
        process.wait()
        
        # 결과 확인
        if os.path.exists(result_json):
            with open(result_json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            return result_video, result_json, json_data
        else:
            st.error(f"JSON 파일이 생성되지 않았습니다.")
            return None, None, None
        
    except Exception as e:
        st.error(f"추론 중 오류 발생: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None


# =========================================================
# 사이드바
# =========================================================
with st.sidebar:
    # 타이틀
    st.markdown('<h1 style="color: #fff; margin-bottom: 12px; margin-top: -75px;">DeepTracer</h1>', unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Drag and drop file here",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        help="Limit 500MB per file",
        key="video_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_video:
        # 임시 파일로 저장
        if st.session_state["video_path"] is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(uploaded_video.read())
                st.session_state["video_path"] = tmp.name
            
            # 업로드된 파일 바이트도 저장 (나중에 표시용)
            uploaded_video.seek(0)
            st.session_state["video_bytes"] = uploaded_video.read()
            
            # 비디오 정보 추출
            fps, total_frames, duration = get_video_info(st.session_state["video_path"])
            st.session_state["fps"] = fps
            st.session_state["total_frames"] = total_frames
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # 추론 실행 버튼
    if uploaded_video and not st.session_state["processing_done"]:
        if st.button("분석 시작", use_container_width=True, type="primary"):
            # 진행률 표시 UI
            with st.sidebar:
                st.markdown("---")
                
                # ▼▼▼ [중요] 이 두 줄이 꼭 있어야 합니다! ▼▼▼
                progress_text = st.empty()   # 글자가 뜰 공간 만들기
                progress_bar = st.progress(0) # 진행바 만들기
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                
                # 2. 로딩 메시지 띄우기 (이제 변수가 있으니 에러가 안 납니다)
                progress_text.markdown("AI 모델 로딩 중... (잠시만 기다려주세요)")
            
            # 출력 디렉토리 생성
            output_dir = tempfile.mkdtemp()
            
            # 추론 실행
            result_video, result_json, json_data = run_inference(
                st.session_state["video_path"], 
                output_dir,
                progress_bar=progress_bar,
                progress_text=progress_text
            )
            
            if result_video and os.path.exists(result_video):
                st.session_state["result_video_path"] = result_video
                st.session_state["json_path"] = result_json
                st.session_state["json_data"] = json_data
                st.session_state["processing_done"] = True
                
                progress_bar.progress(1.0)
                progress_text.text("분석 완료!")
                time.sleep(1)
                st.rerun()
            else:
                progress_text.text("분석 실패")
                st.error("추론 결과를 생성하지 못했습니다.")
    
    # JSON 다운로드 버튼
    if st.session_state["json_data"]:
        # 여백을 위한 얇은 선 (취향껏)
        st.markdown('<div style="border-top: 1px solid #333; margin-top: 10px; margin-bottom: 20px;"></div>', unsafe_allow_html=True)
        
        json_str = json.dumps(st.session_state["json_data"], ensure_ascii=False, indent=4)
        
        # ▼▼▼ [수정] 양옆에 1만큼 여백을 주고, 가운데(2)에 버튼을 놓음 ▼▼▼
        st.markdown('<div style="border-top: ; margin-top: 20px; margin-bottom: 10px;"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1]) 
        with col2:
            st.download_button(
                label="JSON 다운로드",
                data=json_str.encode('utf-8'),
                file_name="plate_detection_result.json",
                mime="application/json",
                use_container_width=True
            )

    
    # 프레임 컨트롤 (분석 완료 후)
    if st.session_state["processing_done"] and st.session_state["json_data"]:
        st.markdown("---")
        st.markdown('<p style="color: #888; font-size: 13px; margin-bottom: 20px;">움직여서 프레임을 선택해보세요</p>', unsafe_allow_html=True)
        
        current_frame = st.slider(
            "프레임",
            min_value=0,
            max_value=max(1, st.session_state["total_frames"] - 1),
            value=st.session_state["current_frame"],
            key="frame_slider",
            label_visibility="collapsed"
        )
        st.session_state["current_frame"] = current_frame
        
        # 현재 프레임 번호판 개수 표시
        current_plates = find_plates_for_frame(st.session_state["json_data"], current_frame)
        if current_plates:
            st.markdown(f'<p style="color: #4a9eff; font-size: 13px;">현재 프레임에서 {len(current_plates)}개 번호판 검출</p>', unsafe_allow_html=True)
        
        c1, prev_col, next_col, c4 = st.columns([0.5, 2, 2, 0.5])
        
        with prev_col:
            # use_container_width=True를 써야 칸에 꽉 차서 예쁩니다
            if st.button("◀  이전 프레임", use_container_width=True):
                st.session_state["current_frame"] = max(0, current_frame - 5)
                st.session_state["current_plate_idx"] = 0
                st.rerun()
                
        with next_col:
            if st.button("다음 프레임  ▶", use_container_width=True):
                st.session_state["current_frame"] = min(st.session_state["total_frames"] - 1, current_frame + 5)
                st.session_state["current_plate_idx"] = 0
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 평가 섹션
    if st.session_state["processing_done"]:
        st.markdown("---")
        st.markdown('<p style="color: #fff; font-size: 22px; text-align: ; font-weight: 500; margin-bottom: 16px;">검출 결과를 평가해주세요</p>', unsafe_allow_html=True)
        
        rating = st.radio(
            "평가",
            options=["좋아요", "평범해요", "별로에요"],
            horizontal=True,
            label_visibility="collapsed"
        )
        
        if st.button("전송", use_container_width=True):
            st.success("소중한 평가 감사합니다")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 초기화 버튼
    if st.session_state["processing_done"]:
        st.markdown("---")
        if st.button("새 영상 분석", use_container_width=False):
            # 세션 초기화
            st.session_state["video_path"] = None
            st.session_state["video_bytes"] = None
            st.session_state["result_video_path"] = None
            st.session_state["json_data"] = None
            st.session_state["json_path"] = None
            st.session_state["current_frame"] = 0
            st.session_state["current_plate_idx"] = 0
            st.session_state["processing_done"] = False
            st.rerun()


# =========================================================
# 메인 컨텐츠
# =========================================================

# 분석 전 상태
if not st.session_state["processing_done"]:
    if not uploaded_video:
        st.markdown("""
        <div style="
            background-color: #121212;
            border-radius: 12px;
            padding: 120px 40px;
            text-align: center;
            border: 1px solid #2a2a2a;
        ">
            <h2 style="color: #fff; margin-bottom: 12px; margin-left: 22px;">  블랙박스 영상 분석 서비스</h2>
            <p style="color: #888; font-size: 20px;">왼쪽에서 영상을 업로드하고 분석을 시작하세요</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="
            background-color: #121212;
            border-radius: 12px;
            padding: 80px 40px;
            text-align: center;
            border: 1px solid #2a2a2a;
        ">
            <p style="color: #4a9eff; font-size: 48px; margin-bottom: 24px;"></p>
            <h3 style="color: #888; margin-bottom: 12px;">영상이 업로드되었습니다</h3>
            <p style="color: #888; font-size: 14px;">왼쪽의 "분석 시작" 버튼을 클릭하세요</p>
        </div>
        """, unsafe_allow_html=True)

# 분석 완료 후 결과 표시
else:
    # 1. 결과 영상 표시
    st.markdown("""
    <div style="
        border-left: 5px solid #4a9eff;
        padding-left: 15px;
        margin-bottom: 20px;
    ">
        <h3 style="color: #fff; margin: 0; font-weight: 700; letter-spacing: 1px;">
            영상 분석 결과
        </h3>
        <p style="color: #888; font-size: 14px; margin: 4px 0 0 0;">
            Video Analysis Result
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 영상 표시 시도
    video_displayed = False
    
    # 결과 영상 시도
    if st.session_state.get("result_video_path") and os.path.exists(st.session_state["result_video_path"]):
        try:
            with open(st.session_state["result_video_path"], 'rb') as f:
                video_bytes = f.read()
            if len(video_bytes) > 0:
                st.video(video_bytes)
                video_displayed = True
        except Exception as e:
            st.warning(f"결과 영상 로드 실패: {e}")
    
    # 결과 영상 실패 시 원본 영상
    if not video_displayed and st.session_state.get("video_bytes"):
        try:
            st.video(st.session_state["video_bytes"])
            video_displayed = True
            st.caption("⚠️ 원본 영상 (결과 영상 로드 실패)")
        except Exception as e:
            st.warning(f"원본 영상 로드 실패: {e}")
    
    if not video_displayed:
        st.warning("영상을 표시할 수 없습니다.")
    
    # 결과 영상 다운로드 버튼
    if st.session_state.get("result_video_path") and os.path.exists(st.session_state["result_video_path"]):
        
        # ▼▼▼ 글꼴/크기 변경 스타일 코드 ▼▼▼
        st.markdown("""
        <style>
        /* 1. 메인 화면 버튼 글자 설정 */
        div[data-testid="stDownloadButton"] button p {
            font-size: 20px !important;              /* 글자 크기 (원하는대로 조절) */
            font-family: sans-serif !important;
            font-weight: 700 !important;             /* 글자 굵기 (Bold) */
            color: #ffffff !important;               /* 글자 색상 */
        }
        
        /* 1-1. 버튼 껍데기도 글자에 맞춰 좀 키워줌 */
        div[data-testid="stDownloadButton"] button {
            padding: 10px 20px !important;           /* 버튼 내부 여백 (위아래, 좌우) */
            height: auto !important;
        }

        /* 2. [방어] 사이드바 버튼은 원래대로 돌려놓기 */
        section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button p {
            font-size: 14px !important;
            text-align: center;
            font-family: sans-serif !important;
            font-weight: 500 !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stDownloadButton"] button {
            padding: 8px 16px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

        with open(st.session_state["result_video_path"], 'rb') as f:
            st.download_button(
                label="결과 영상 다운로드",
                data=f.read(),
                file_name="result_video.mp4",
                mime="video/mp4",
                use_container_width=False # 버튼 꽉 차게 하려면 이거 유지하세요
            )
    
    st.markdown("---")
    
    # 2. 프레임별 번호판 뷰어
    st.markdown("""
    <div style="
        border-left: 5px solid #4a9eff;
        padding-left: 15px;
        margin-bottom: 20px;
        margin-top: 40px; /* 위쪽 다운로드 버튼이랑 간격 좀 띄웠습니다 */
    ">
        <h3 style="color: #fff; margin: 0; font-weight: 700; letter-spacing: 1px;">
            프레임별 번호판 검출 결과
        </h3>
        <p style="color: #888; font-size: 14px; margin: 4px 0 0 0;">
            Frame-by-Frame Detection Details
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 현재 프레임 정보
    current_frame = st.session_state["current_frame"]
    fps = st.session_state["fps"]
    current_time = current_frame / fps if fps > 0 else 0
    
    # 현재 프레임의 번호판 찾기
    current_plates = find_plates_for_frame(st.session_state["json_data"], current_frame)
    
    # 레이아웃: 프레임 이미지 | 번호판 정보
    col_frame, col_plate = st.columns([1.2, 1])
    
    with col_frame:
        # 프레임 이미지 표시
        video_path = st.session_state["result_video_path"] or st.session_state["video_path"]
        if video_path and os.path.exists(video_path):
            frame_img = get_frame_at_index(video_path, current_frame)
            if frame_img is not None:
                st.image(frame_img, use_container_width=True)
                st.caption(f"프레임 {current_frame} | {format_time(current_time)}")
            else:
                st.warning("프레임을 불러올 수 없습니다.")
    
    with col_plate:
        if current_plates:
            # 번호판 인덱스
            plate_idx = st.session_state["current_plate_idx"]
            if plate_idx >= len(current_plates):
                plate_idx = 0
                st.session_state["current_plate_idx"] = 0
            
            plate = current_plates[plate_idx]
            
            # 인덱스 표시
            st.markdown(f'<span class="plate-index">{plate_idx + 1}번째 번호판</span>', unsafe_allow_html=True)
            
            # 번호판 이미지
            img_base64 = plate.get("image_base64", plate.get("crop_image", None))
            if img_base64:
                img = decode_base64_image(img_base64)
                if img:
                    st.image(img, use_container_width=True)
            
            # 검출 정보
            plate_number = plate.get("plate_number", plate.get("plate", plate.get("ocr_result", "N/A")))
            ocr_conf = plate.get("ocr_confidence", plate.get("confidence", plate.get("ocr_confidence_score", 0)))
            det_conf = plate.get("det_confidence", plate.get("detection_confidence", None))
            
            conf_color = get_confidence_color_hex(ocr_conf)
            
            st.markdown(f"""
            <div style="margin-top: 16px;">
                <p style="color: #888; font-size: 13px; margin-bottom: 4px;">검출 내용</p>
                <p style="
                    color: #ffffff;
                    font-size: 30px;       /* 글자 크기: 여기서 조절하세요 */
                    font-weight: 900;      /* 굵기: 아주 두껍게 */
                    letter-spacing: 2px;   /* 자간: 글자 사이 간격 */
                    line-height: 1.1;      /* 줄 간격 */
                    margin: 0;
                    text-shadow: 0 0 10px rgba(255, 255, 255, 0.3); /* 살짝 빛나는 효과 */
                ">{plate_number}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="margin-top: 8px;">
                <p style="color: #888; font-size: 13px; margin-bottom: 4px;">검출 신뢰도</p>
                <p style="color: {conf_color}; font-size: 20px; font-weight: 600;">{ocr_conf*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # bbox 크기
            bbox = plate.get("bbox", plate.get("coordinate", None))
            if bbox:
                if isinstance(bbox[0], list):
                    w = bbox[1][0] - bbox[0][0]
                    h = bbox[1][1] - bbox[0][1]
                else:
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                st.markdown(f"""
                <div style="margin-top: 12px;">
                    <p style="color: #888; font-size: 13px; margin-bottom: 4px;">이미지 크기</p>
                    <p style="color: #fff; font-size: 16px;">({w}, {h})</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 번호판 네비게이션 버튼 (여러 개일 때)
            if len(current_plates) > 1:
                st.markdown("<br>", unsafe_allow_html=True)
                nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
                
                with nav_col1:
                    if st.button("◀ 이전 번호판", use_container_width=True, disabled=(plate_idx == 0)):
                        st.session_state["current_plate_idx"] = plate_idx - 1
                        st.rerun()
                
                with nav_col2:
                    st.markdown(f"""
                    <p style="text-align: center; color: #888; font-size: 18px; padding-top: 5px;">
                        {plate_idx + 1} / {len(current_plates)}
                    </p>
                    """, unsafe_allow_html=True)
                
                with nav_col3:
                    if st.button("다음 번호판 ▶", use_container_width=True, disabled=(plate_idx >= len(current_plates) - 1)):
                        st.session_state["current_plate_idx"] = plate_idx + 1
                        st.rerun()
        
        else:
            # 번호판이 없는 경우 (else 문 내부)
            st.markdown("""
            <div class="plate-card" style="text-align: center; padding: 80px 20px;">
                <div style="margin-bottom: 24px; opacity: 0.7;">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#ccc" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                </div>
                <p style="color: #ccc; font-size: 24px; font-weight: 700; margin: 0;">해당 프레임 검출 결과 없음</p>
            </div>
            """, unsafe_allow_html=True)
    
    # 통계 요약
    if st.session_state["json_data"]:
        st.markdown("---")
        
        # 1. 디자인 (제목)
        st.markdown("""
        <div style="
            border-left: 5px solid #4a9eff;
            padding-left: 15px;
            margin-bottom: 20px;
            margin-top: 20px;
        ">
            <h3 style="color: #fff; margin: 0; font-weight: 700; letter-spacing: 1px;">
                검출 통계 리포트
            </h3>
            <p style="color: #888; font-size: 14px; margin: 4px 0 0 0;">
                Statistical Analysis & Summary
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        json_data = st.session_state["json_data"]
        
        # 2. 데이터 계산
        total_detections = len(json_data)  # 총 개수
        
        # 고유 번호판 개수 (중복 제거)
        unique_plates = len(set(
            p.get("plate_number", p.get("plate", p.get("ocr_result", "unknown"))) 
            for p in json_data
        ))
        
        # 3. 결과 표시 (신뢰도 빼고 3칸으로 변경)
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        
        with stat_col1:
            st.metric("총 검출 횟수", f"{total_detections}건")
        with stat_col2:
            st.metric("고유 번호판", f"{unique_plates}개")
        with stat_col3:
            st.metric("분석 프레임", f"{st.session_state['total_frames']}개")


# =========================================================
# 푸터
# =========================================================
st.markdown("""
<div style="
    text-align: center;
    padding: 24px;
    color: #888;
    font-size: 18px;
    font-weight: 500;
    margin-top: 40px;
">
    DeepTracer | Powered by YOLOv8n + LPRNet
</div>
""", unsafe_allow_html=True)