import os
import cv2
import glob
from pathlib import Path
from tqdm import tqdm


def extract_frames(
    input_dir: str = "../Dataset/LEMON",
    output_dir: str = "../Dataset/LEMON_frames",
    fps: int = 1,
):
    """
    MP4 파일에서 프레임을 추출하여 저장합니다.
    
    Args:
        input_dir  : MP4 파일들이 있는 폴더 경로
        output_dir : 이미지 저장 폴더 경로
        fps        : 저장할 프레임 속도 (기본값: 1fps)
    """
    # MP4 파일 목록 수집 (하위 폴더 포함)
    mp4_files = sorted(glob.glob(os.path.join(input_dir, "**/*.mp4"), recursive=True))
    mp4_files += sorted(glob.glob(os.path.join(input_dir, "*.mp4")))
    mp4_files = sorted(set(mp4_files))  # 중복 제거

    if not mp4_files:
        print(f"[ERROR] '{input_dir}' 경로에서 MP4 파일을 찾을 수 없습니다.")
        return

    print(f"총 {len(mp4_files)}개의 MP4 파일 발견")
    os.makedirs(output_dir, exist_ok=True)

    success_count = 0
    fail_count = 0

    for video_idx, video_path in enumerate(tqdm(mp4_files, desc="비디오 처리 중")):
        # 비디오 폴더명: 0000, 0001, ...
        folder_name = f"{video_idx:04d}"
        save_dir = os.path.join(output_dir, folder_name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"\n[WARN] 열 수 없는 파일 건너뜀: {video_path}")
            fail_count += 1
            continue

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            print(f"\n[WARN] FPS 정보 없음, 건너뜀: {video_path}")
            cap.release()
            fail_count += 1
            continue

        # 몇 프레임마다 저장할지 계산
        frame_interval = max(1, round(video_fps / fps))

        frame_num = 0    # 읽은 원본 프레임 번호
        save_num = 0     # 저장된 이미지 번호

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % frame_interval == 0:
                img_name = f"{save_num:05d}.jpg"
                img_path = os.path.join(save_dir, img_name)
                cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                save_num += 1

            frame_num += 1

        cap.release()
        success_count += 1

    print(f"\n완료!")
    print(f"  성공: {success_count}개 / 실패(건너뜀): {fail_count}개")
    print(f"  저장 경로: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MP4 → 1fps 이미지 추출기")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../Dataset/LEMON",
        help="MP4 파일들이 있는 폴더 (기본값: Dataset/LEMON)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../Dataset/LEMON_frames",
        help="이미지 저장 폴더 (기본값: Dataset/LEMON_frames)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="저장할 FPS (기본값: 1)",
    )
    args = parser.parse_args()

    extract_frames(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
    )