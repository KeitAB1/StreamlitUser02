import os
from datetime import datetime
import streamlit as st
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import imagehash
import shutil
import matplotlib.pyplot as plt
from auxiliary import Rec_utils as ru


class ImgRec:
    def __init__(self):
        self.reader = None
        self.average_confidences = []
        self.x1,self.y1, = 0,0
        self.x2,self.y2 = 160,90
        self.mask_type = 'black'
        self.isMask = False

    def clear_confidences(self):
        # 清空置信度列表
        self.average_confidences = []

    def set_reader(self, reader_instance):
        """
        设置模型实例。
        """
        self.reader = reader_instance

    #在图像上绘制方框及识别的文本
    def draw_boxes(self, image, results):

        for (bbox, text, prob) in results:
            # 获取边框的坐标
            top_left = tuple([int(val) for val in bbox[0]])
            bottom_right = tuple([int(val) for val in bbox[2]])

            # 绘制矩形方框
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

            # 在框旁边写入识别到的文本及置信度
            cv2.putText(image, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        return image

    #遮掩参数设置
    def mask_Settings(self):
        # 创建一个开关按钮
        toggle = st.sidebar.toggle("遮掩开关")

        # 根据开关状态显示不同内容
        if toggle:
            self.isMask = True
            # 使用 Streamlit 的 slider 来选择遮掩区域
            self.x1 = st.sidebar.number_input("左上角x坐标", min_value=0, value=self.x1, step=1)
            self.y1 = st.sidebar.number_input("左上角y坐标", min_value=0, value=self.y1, step=1)
            self.x2 = st.sidebar.number_input("右下角x坐标", min_value=0, value=self.x2, step=1)
            self.y2 = st.sidebar.number_input("右下角y坐标", min_value=0, value=self.y2, step=1)
            # 选择遮掩类型
            self.mask_type = st.sidebar.selectbox("选择遮掩方式", ["black", "blur", "remove"])

            st.sidebar.write("遮掩已打开")
        else:
            self.isMask = False
            st.sidebar.write("遮掩已关闭")


    #参数设置
    def Parameter_Settings(self):
        self.link_threshold = st.sidebar.number_input("控制相邻字符之间的连接度阈值", min_value=0, max_value=1, value=0.8, step=0.05)

    # 进行OCR识别
    def Rec_fun(self, image, file_name, IMAGE_SAVE_DIR):
        '''
        传入：图像，图像文件名，图像输出路径
        返回：图像文件名，识别编码，识别完成时间，平均识别准确度
        '''

        # 转换为NumPy数组
        image = np.array(image)
        if self.isMask:
            # 获取图像尺寸
            img_height, img_width = image.shape[:2]
            image = mask_region(image, range_constrained(self.x1,0,img_width), range_constrained(self.y1,0,img_height), \
                                range_constrained(self.x2,0,img_width), range_constrained(self.y2,0,img_height), self.mask_type)

        # image = cv2.resize(image, None, fx=10, fy=10, interpolation=cv2.INTER_CUBIC)
        # _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # # 检查图像维度，确保是三通道图像
        if len(image.shape) == 2:  # 如果是灰度图像，转为RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # # 确保传入的图像是灰度图
        # if len(image.shape) == 3 and image.shape[2] == 3:  # 判断是否是彩色图像
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像

        # # 进行阈值处理
        # _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # 文字识别
        allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()-/*'
        results = self.reader.readtext(image, allowlist=allowlist, link_threshold=0.8, paragraph=False)
        detected_text = []
        recognition_text = ''
        total_confidence = 0.0  # 记录总置信度
        for (bbox, text, prob) in results:
            recognition_text = recognition_text + text + ' '
            detected_text.append(f"{text} (Confidence: {prob:.2f})")
            total_confidence += prob  # 累加每次识别的置信度

        # 计算平均置信度
        average_confidence = total_confidence / len(results) if results else 0.0

        # 终端输出结果
        print(f'Recognition text: {recognition_text}')

        # 在图像上绘制方框及识别的文本
        image_with_boxes = self.draw_boxes(image, results)
        # 转换为PIL Image显示
        image_with_boxes = Image.fromarray(image_with_boxes)
        # 保存处理过的图片到指定目录
        image_with_boxes.save(os.path.join(IMAGE_SAVE_DIR, file_name))  # 保存调整后的图像
        # 保存处理完成时间
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return file_name, recognition_text, timestamp, average_confidence


    #文件夹图片识别
    def process_images_from_folder(self, folder_path, progress_placeholder, IMAGE_SAVE_DIR):
        """对文件夹中的所有图像进行OCR识别并返回结果，加入图片校正和调整过程"""
        data = []
        #读取图片目录
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png','.bmp'))]
        total_images = len(image_files)

        if total_images == 0:
            return None, 0  # 如果文件夹中没有图片，返回None

        #依次处理文件夹中图片
        self.clear_confidences()
        for idx, file_name in enumerate(image_files):
            image_path = os.path.join(folder_path, file_name)   #路径
            image = Image.open(image_path)  #打开图片
            file_name,recognition_text,timestamp,average_confidence = self.Rec_fun(image, file_name, IMAGE_SAVE_DIR)
            self.average_confidences.append(average_confidence)
            data.append({"Filename": file_name, "Recognized Text": recognition_text, "Average Confidence": average_confidence,"Timestamp": timestamp})

            # 更新进度条
            progress_placeholder.progress((idx + 1) / total_images)

        return data, total_images


    # 上传图片识别
    def process_uploaded_images(self, uploaded_files, progress_placeholder, IMAGE_SAVE_DIR):
        """处理上传的图片并返回识别结果"""
        data = []  # 保存识别结果的列表
        total_files = len(uploaded_files)  # 上传文件总数

        if total_files == 0:
            return None, 0  # 如果还为上传图片，返回None

        # 遍历每个上传的文件
        self.clear_confidences()
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)  # 打开图片
            file_name, recognition_text, timestamp, average_confidence = self.Rec_fun(image, uploaded_file.name, IMAGE_SAVE_DIR)
            self.average_confidences.append(average_confidence)
            data.append({"Filename": file_name, "Recognized Text": recognition_text, "Average Confidence": average_confidence,"Timestamp": timestamp})

            # 更新进度条
            progress_placeholder.progress((idx + 1) / total_files)
        return data  # 返回识别数据列表


    def Image_Recongnotion(self, IMAGE_SAVE_DIR, CSV_FILE_PATH):
        st.header("Image Code Recognition with Bounded Boxes")
        st.write("Select Image Input Method")

        # 选择图像输入方式
        option = st.selectbox('Please choose', ['Images from project folder', 'Manual upload'], key="key_for_ImgRec_kinds")

        if option == 'Images from project folder':
            base_folder_path = 'data/plate_img'
            subfolders = [f for f in os.listdir(base_folder_path) if
                          os.path.isdir(os.path.join(base_folder_path, f)) and f.startswith('Image_src')]

            if subfolders:
                selected_subfolder = st.selectbox('Please select an image folder', subfolders, key="key_for_ImgRec_folder")
                folder_path = os.path.join(base_folder_path, selected_subfolder)

                #显示文件夹中图片
                image_files = os.listdir(folder_path)
                if image_files:
                    selected_image = st.selectbox("Select an image to view", image_files,key="key_for_preview_image")
                    image_path = os.path.join(folder_path, selected_image)
                    image = Image.open(image_path)
                    # 在侧边栏中显示图片
                    st.image(image, caption=os.path.basename(image_path), use_column_width=True)

                if st.button('Start Recognition'):
                    if os.path.exists(folder_path):
                        progress_placeholder = st.empty()
                        # 进行识别
                        data, total_images = self.process_images_from_folder(folder_path, progress_placeholder,
                                                                        IMAGE_SAVE_DIR)

                        if total_images == 0:
                            st.warning(f'No images found in folder {selected_subfolder}!')
                        elif data:
                            ru.append_to_csv(data, CSV_FILE_PATH)
                            df = pd.DataFrame(data)
                            st.dataframe(df)  # 实时显示当前处理的图像结果
                            st.success(
                                f'Recognition complete! Results saved to recognized_results.csv (Folder: {selected_subfolder})')
                    else:
                        st.error(f'Folder {folder_path} does not exist!')

        elif option == 'Manual upload':
            uploaded_files = st.file_uploader('Upload image files', type=['jpg', 'png', 'bmp'],
                                              accept_multiple_files=True)
            if uploaded_files:
                if st.button('Start Recognition'):
                    progress_placeholder = st.empty()
                    data = self.process_uploaded_images(uploaded_files, progress_placeholder, IMAGE_SAVE_DIR)
                    if data:
                        ru.append_to_csv(data, CSV_FILE_PATH)
                        df = pd.DataFrame(data)
                        st.dataframe(df)  # 实时显示当前处理的图像结果
                        st.success('Recognition complete! Results saved to recognized_results.csv')

        # # 显示识别结果csv表格
        # csv_display(CSV_FILE_PATH)
        #
        # # 侧边栏显示历史识别图片
        # Rec_history_image(IMAGE_SAVE_DIR)

    #从视频中截取帧，保存并去重
    def extract_unique_frames_from_video(self, frame_interval,video_path, frames_cache_folder, final_frames_folder):
        # 创建文件夹（如果不存在）
        ru.ensure_directory_exists(frames_cache_folder)
        ru.ensure_directory_exists(final_frames_folder)
        #清除文件夹内容
        ru.clear_folder(frames_cache_folder)
        ru.clear_folder(final_frames_folder)


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
        st.success(f"去重完成，共保留 {len(hashes)} 张图像。")
        print(f"去重完成，共保留 {len(hashes)} 张图像。")


    def Video_Recognition(self, IMAGE_SAVE_DIR, CSV_FILE_PATH):
        # 设置文件路径
        video_folder = "data/plate_video"
        frames_cache_folder = "data/video_frames/frames_cache"
        final_frames_folder = 'data/video_frames/final_frames'


        st.header("视频文本识别")
        st.write("请选择图像输入方式")

        # 确定保存间隔
        frame_interval = st.number_input("每隔多少帧保存一次图像", min_value=1, value=20, step=1)

        option = st.selectbox('请选择输入方式', ['项目文件夹中的视频'], key="key_for_VidRec_kinds")

        if option == '项目文件夹中的视频':
            video_folder = 'data/plate_video'
            videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

            if videos:
                selected_video = st.selectbox('请选择视频文件', videos, key="key_for_VidRec_file")

                if st.button("开始识别"):
                    ru.ensure_directory_exists(video_folder)
                    video_path = os.path.join(video_folder, selected_video)
                    self.extract_unique_frames_from_video(frame_interval ,video_path, frames_cache_folder, final_frames_folder)

                    progress_placeholder = st.empty()
                    # 进行识别
                    data, total_images = self.process_images_from_folder(final_frames_folder, progress_placeholder,
                                                                         IMAGE_SAVE_DIR)

                    if total_images == 0:
                        st.warning(f'No images found in folder !')
                    elif data:
                        ru.append_to_csv(data, CSV_FILE_PATH)
                        df = pd.DataFrame(data)
                        st.dataframe(df)  # 实时显示当前处理的图像结果
                        st.success(
                            f'Recognition complete! Results saved to recognized_results.csv ')

            else:
                st.write("项目文件夹中没有找到视频文件。")

            # # 显示识别结果csv表格
            # csv_display(CSV_FILE_PATH)
            #
            # # 侧边栏显示历史识别图片
            # Rec_history_image(IMAGE_SAVE_DIR)


# 创建全局实例
img_rec_instance = ImgRec()


def csv_display(CSV_FILE_PATH):
    st.write("Current Content in CSV File")

    # 清除识别结果(csv表格)
    if st.button('Clear CSV File'):
        with st.spinner('Clearing CSV file...'):
            try:
                ru.clear_csv(CSV_FILE_PATH)
                st.success('CSV file content cleared')
            except Exception as e:
                st.error(f"Error clearing CSV file: {e}")

    # 显示识别结果(csv表格)
    if os.path.exists(CSV_FILE_PATH):
        if ru.is_csv_empty(CSV_FILE_PATH):
            st.write('No recognition data available')
        else:
            try:
                df = pd.read_csv(CSV_FILE_PATH)
                st.dataframe(df)
            except pd.errors.EmptyDataError:
                st.error('CSV file is empty or cannot be parsed.')
            except pd.errors.ParserError as e:
                st.error(f"Error parsing CSV file: {e}")
            except Exception as e:
                st.error(f"An error occurred while reading the CSV file: {e}")
    else:
        st.warning('CSV file does not exist.')

def Rec_history_image(IMAGE_SAVE_DIR):
    # 侧边栏显示历史识别图片
    st.sidebar.title("Recognized Image History")

    if st.sidebar.button('Clear Image History'):
        ru.clear_folder(IMAGE_SAVE_DIR)
        st.sidebar.success("Image history cleared")

    image_files = os.listdir(IMAGE_SAVE_DIR)
    if image_files:
        selected_image = st.sidebar.selectbox("Select an image to view", image_files, key="key_for_history_image")
        image_path = os.path.join(IMAGE_SAVE_DIR, selected_image)
        image = Image.open(image_path)
        # 在侧边栏中显示图片
        st.sidebar.image(image, caption=os.path.basename(image_path), use_column_width=True)
        # display_image_with_rotation(image_path)  # 使用新方法显示图片


    else:
        st.sidebar.write('No recognized image history available')


def plot_confidences(average_confidences):
    # 绘制柱状图
    st.write("OCR 识别平均置信度柱状图")
    if average_confidences:
        # 创建柱状图
        fig, ax = plt.subplots()
        ax.bar(range(len(average_confidences)), average_confidences)
        ax.set_xlabel('OCR Call Number')
        ax.set_ylabel('Average Confidence')
        ax.set_title('Average Confidence per OCR Call')
        st.pyplot(fig)
    else:
        st.write("没有可用的识别置信度数据")


# 定义遮掩函数
def mask_region(image, x1, y1, x2, y2, mask_type="black"):
    # 复制图像
    masked_image = image.copy()

    # 检查区域是否有效
    if x1 >= x2 or y1 >= y2:
        st.error("无效的遮掩区域，请确保左上角坐标小于右下角坐标")
        return masked_image

    # 判断使用哪种遮掩类型
    if mask_type == "black":
        cv2.rectangle(masked_image, (x1, y1), (x2, y2), (0, 0, 0), -1)  # 黑色遮掩
    elif mask_type == "blur":
        roi = masked_image[y1:y2, x1:x2]
        roi = cv2.GaussianBlur(roi, (23, 23), 30)
        masked_image[y1:y2, x1:x2] = roi  # 模糊处理
    elif mask_type == "remove":
        masked_image[y1:y2, x1:x2] = [255, 255, 255]  # 去除为白色

    return masked_image


def range_constrained(num, minVal, maxVal):
    if num < minVal:
        num = 0
    if num > maxVal:
        num = maxVal
    return num


