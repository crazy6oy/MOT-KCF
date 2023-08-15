import cv2
import tqdm


def merge_video_by_opencv(videos_path, save_path):
    videos_cap = [cv2.VideoCapture(x) for x in videos_path]
    fps = videos_cap[0].get(cv2.CAP_PROP_FPS)
    H = int(videos_cap[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(videos_cap[0].get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(save_path, fourcc, fps, (W, H))

    merge_log = ''
    for video_path, video_cap in zip(videos_path, videos_cap):
        merge_log += f"{video_path}\n"
        loss_frames_num = []
        frames_count = int(round(video_cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        for i in tqdm.tqdm(range(frames_count)):
            ret, frame = video_cap.read()
            if ret == False:
                loss_frames_num.append(i)
                continue
            else:
                writer.write(frame)

        merge_log += f"loss: {loss_frames_num}\n"

    writer.release()

    return save_path
