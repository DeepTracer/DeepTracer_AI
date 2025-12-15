from torch.utils.data import *
import cv2
import numpy as np
import random
import os

# 한국 자동차 번호판용 문자셋 (지역명, 배 등 포함)
CHARS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '가', '나', '다', '라', '마', '거', '너', '더', '러', '머', '버', '서', '어', '저',
    '고', '노', '도', '로', '모', '보', '소', '오', '조',
    '구', '누', '두', '루', '무', '부', '수', '우', '주',
    '아', '바', '사', '자', '하', '허', '호', '윤',
    '경', '기', '강', '원', '충', '북', '남', '전', 
    '제', '인', '천', '광', '대', '울', '산', '세', '종',
    '배', '_'
]

CHARS_DICT = {char:i for i, char in enumerate(CHARS)}

class LPRDataLoader(Dataset):
    def __init__(self, img_dirs, imgSize, lpr_max_len=None, is_train=True):
        self.img_dirs = img_dirs
        self.img_paths = []
        self.img_size = imgSize
        self.lpr_max_len = lpr_max_len
        self.is_train = is_train
        
        # 리스트 안에 있는 폴더들을 하나씩 꺼내서 확인
        for img_dir in self.img_dirs:
            if os.path.isdir(img_dir):
                print(f"폴더 읽는 중: {img_dir} ...")
                all_files = os.listdir(img_dir)
                for f in all_files:
                    # 이미지 확장자 체크
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        self.img_paths.append(os.path.join(img_dir, f))
            else:
                print(f"경로를 찾을 수 없음 (건너뜀): {img_dir}")

        print(f"총 {len(self.img_paths)}장의 이미지를 로드했습니다.")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        filename = self.img_paths[index]
        
        # 1. 이미지 읽기 (한글 경로 지원)
        try:
            with open(filename, "rb") as f:
                bytes = bytearray(f.read())
                numpy_array = np.asarray(bytes, dtype=np.uint8)
                Image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            return self.__getitem__(np.random.randint(self.__len__()))

        if Image is None:
            return self.__getitem__(np.random.randint(self.__len__()))

        height, width, _ = Image.shape
        if height != self.img_size[1] or width != self.img_size[0]:
            Image = cv2.resize(Image, self.img_size)
        
        Image = self.transform(Image)

        # 2. 라벨 파싱 (한국 번호판: 파일명 = 라벨)
        # 예: "./data/train/12가3456.jpg" -> "12가3456"
        basename = os.path.basename(filename)
        label_str = os.path.splitext(basename)[0]
        
        # 공백 제거 및 문자열 정리
        label_str = label_str.replace(' ', '').split('-')[0].split('_')[0]

        label = []
        for c in label_str:
            if c in CHARS_DICT:
                label.append(CHARS_DICT[c])
            else:
                pass

        # 라벨 길이가 너무 짧으면(데이터 불량) 다른 이미지로 대체
        if len(label) == 0:
            return self.__getitem__(np.random.randint(self.__len__()))

        return Image, label, len(label)

    def transform(self, img):
        # [수정 2] self.is_train이 True일 때만(학습 중일 때만) 데이터 증강 적용
        if self.is_train and random.random() < 0.5:
            # 1. 저해상도 시뮬레이션
            if random.random() < 0.5:
                h, w, _ = img.shape
                scale = random.uniform(0.5, 0.9) 
                small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # 2. 블러
            if random.random() < 0.5:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
                
            # 3. 노이즈 및 밝기
            img = img.astype(np.float32)
            noise = np.random.randn(*img.shape) * random.randint(3, 15)
            img += noise
            img = np.clip(img, 0, 255)

        img = img.astype('float32')
        img -= 127.5
        img *= 0.0078125
        img = np.transpose(img, (2, 0, 1))
        return img