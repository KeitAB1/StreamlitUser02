import streamlit as st
import easyocr
import cv2
import os
import csv
import pandas as pd
from auxiliary import Rec_utils as ru
import streamlit as st
import cv2
import os
from PIL import Image
import imagehash
import shutil
import time


# 设置文件路径
video_folder = "data/plate_video"
frames_cache_folder = "data/video_frames/frames_cache"
final_frames_folder = 'data/video_frames/final_frames'


# 创建文件夹（如果不存在）
ru.ensure_directory_exists(video_folder)
ru.ensure_directory_exists(frames_cache_folder)
ru.ensure_directory_exists(final_frames_folder)

def Video_Recognition():
    st.header("视频文本识别")
    st.write("请选择图像输入方式")

    # 确定保存间隔
    frame_interval = st.number_input("每隔多少帧保存一次图像", min_value=1, value=20, step=1)

    option = st.selectbox('请选择输入方式', ['项目文件夹中的视频'])

    if option == '项目文件夹中的视频':
        video_folder = 'data/plate_video'
        videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

        if videos:
            selected_video = st.selectbox('请选择视频文件', videos)

            if st.button("开始识别"):


                video_path = os.path.join(video_folder, selected_video)
                cap = cv2.VideoCapture(video_path)

                frame_count = 0
                saved_frame_count = 0

                # 读取视频帧并保存图像
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        frame_filename = os.path.join(frames_cache_folder, f"frame_{frame_count}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        saved_frame_count += 1

                    frame_count += 1

                cap.release()
                st.success(f"已保存{saved_frame_count}张图像到 {frames_cache_folder} 文件夹中。")

                # 显示保存的部分图像
                st.write("以下是保存的部分图像：")
                images = os.listdir(frames_cache_folder)
                images.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))  # 根据帧号排序


                # 设置哈希容差，容差越小，相似度要求越高
                hash_tolerance = 5
                hashes = []

                # 遍历源文件夹中的所有图像
                for filename in sorted(os.listdir(frames_cache_folder)):
                    file_path = os.path.join(frames_cache_folder, filename)
                    if filename.endswith('.jpg') or filename.endswith('.png'):
                        image = Image.open(file_path)

                        # 计算图像的感知哈希值
                        img_hash = imagehash.phash(image)

                        # 检查哈希列表中是否存在相似的图像
                        if all(abs(img_hash - existing_hash) > hash_tolerance for existing_hash in hashes):
                            # 如果没有相似图像，将哈希值加入列表
                            hashes.append(img_hash)
                            # 保存该图像到目标文件夹
                            shutil.copy(file_path, os.path.join(final_frames_folder, filename))
                            print(f"保留图像: {filename}")

                print(f"去重完成，共保留 {len(hashes)} 张图像。")

                # 将图像显示在页面中
                for img_name in images[:10]:  # 显示前10张
                    img_path = os.path.join(frames_cache_folder, img_name)
                    image = Image.open(img_path)
                    st.image(image, caption=img_name, use_column_width=True)

        else:
            st.write("项目文件夹中没有找到视频文件。")


