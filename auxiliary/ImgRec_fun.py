import os
import cv2
import time
import shutil
import imagehash
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.graph_objects as go
from auxiliary import Rec_utils as ru
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

class ImgRec:
    def __init__(self):
        self.path_Set()
        self.reader = None
        self.allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789()-/* '
        self.average_confidences = []
        self.Batch = 0
        self.x1,self.y1 = 0,0
        self.x2,self.y2 = 160,90
        self.mask_type = 'black'
        self.isMask = False
        self.isDisplay = False
        self.Rec_df = None

    #初始化路径
    def path_Set(self):
        self.IMAGE_SAVE_DIR = 'result/ImageRecognition_Img' #图片保存路径
        self.CSV_FILE_DIR = 'result/ImageRecognition_CSV'   #csv结果文件夹路径
        self.CSV_FILE_PATH = self.CSV_FILE_DIR + '/recognized_results.csv'  #编码识别csv路径
        self.CSV_OUTPUT_PATH = self.CSV_FILE_DIR + '/Output_steel_data.csv' #编码分割csv路径

    def set_reader(self, reader_instance):
        """
        设置模型实例。
        """
        self.reader = reader_instance

    def clear_confidences(self):
        # 清空置信度列表
        self.average_confidences = []


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
    def Rec_fun(self, image, file_name, IMAGE_SAVE_DIR, correct_text=None):
        '''
        传入：图像，图像文件名，图像输出路径，正确编码
        返回：图像文件名，识别编码，识别完成时间，平均识别准确度，准确率
        '''
        # 如果图像是PIL对象，转换为numpy数组
        if isinstance(image, Image.Image):
            image = np.array(image)

        # 文字识别
        results = self.reader.readtext(image, allowlist=self.allowlist, link_threshold=0.8, paragraph=False)

        # 提取识别结果
        recognition_text = ''
        total_confidence = 0.0
        for (bbox, text, prob) in results:
            recognition_text += text + ' '
            total_confidence += prob

        recognition_text = ru.process_steel_code(recognition_text)
        average_confidence = total_confidence / len(results) if results else 0.0

        accuracy = 0
        if correct_text is not None:
            # 计算准确率
            accuracy = calculate_accuracy(recognition_text, correct_text) if correct_text else None
        accuracy = "{:.2%}".format(accuracy)

        # 保存处理完成时间
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 返回附带准确率的结果
        return file_name, recognition_text, average_confidence, accuracy, timestamp

    import os
    from datetime import datetime, timedelta
    from PIL import Image

    #文件夹图像识别
    def process_images_from_folder(self, folder_path, progress_placeholder, IMAGE_SAVE_DIR, table_data=None):
        """对文件夹中的所有图像进行OCR识别并返回结果，加入图片校正和调整过程"""
        data = []
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.bmp'))]
        total_images = len(image_files)

        if total_images == 0:
            return None, 0  # 如果文件夹中没有图片，返回None

        # 清空置信度列表
        self.clear_confidences()
        self.Batch += 1

        # 获取当前时间和交付时间
        entry_time = datetime.now().strftime('%Y-%m-%d')  # 当前日期
        delivery_time = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')  # 30天后的日期
        Batch = 'Q' + str(self.Batch)

        for idx, file_name in enumerate(image_files):
            image_path = os.path.join(folder_path, file_name)
            image = Image.open(image_path)

            correct_text = None
            if table_data is not None:
                # 查找该图像对应的正确编码
                correct_text = table_data.loc[table_data['Filename'] == file_name, 'Recognized Text'].values[0]

            # 识别图像并计算准确率
            file_name, recognition_text, average_confidence, accuracy, timestamp = self.Rec_fun(image, file_name, IMAGE_SAVE_DIR, correct_text)
            self.average_confidences.append(average_confidence)

            # 将数据追加到data中
            data.append(
                {"Filename": file_name, "Recognized Text": recognition_text, "Average Confidence": average_confidence,
                 "Accuracy": accuracy, "Timestamp": timestamp,
                 "Entry Time": entry_time, "Delivery Time": delivery_time, "Batch": Batch})

            # 更新进度条
            progress_placeholder.progress((idx + 1) / total_images)

        # plot_confidences(self.average_confidences)
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
        self.Batch += 1

        # 获取当前时间和交付时间
        entry_time = datetime.now().strftime('%Y-%m-%d')  # 当前日期
        delivery_time = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')  # 30天后的日期

        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)  # 打开图片
            # 识别图像并计算准确率
            file_name, recognition_text, average_confidence, accuracy, timestamp = self.Rec_fun(image, uploaded_file.name, IMAGE_SAVE_DIR)
            self.average_confidences.append(average_confidence)

            # 将数据追加到data中
            data.append(
                {"Filename": file_name, "Recognized Text": recognition_text, "Average Confidence": average_confidence,
                 "Accuracy": accuracy, "Timestamp": timestamp,
                 "Entry Time": entry_time, "Delivery Time": delivery_time, "Batch": self.Batch})

            # 更新进度条
            progress_placeholder.progress((idx + 1) / total_files)
        return data  # 返回识别数据列表


    def Image_Recongnotion(self, IMAGE_SAVE_DIR, CSV_FILE_PATH):
        st.header("🖼️ 图像编码识别")
        st.write("请选择图像输入方式 📥")

        # 选择图像输入方式
        op1, op2 = '测试数据集', '手动上传图像'
        option = st.selectbox('🔍 请选择输入方式', [op1, op2], key="key_for_ImgRec_kinds")

        if option == op1:
            col_folder, col_file = st.columns([0.5, 0.2])
            base_folder_path = 'data/plate_img'
            subfolders = [f for f in os.listdir(base_folder_path) if
                          os.path.isdir(os.path.join(base_folder_path, f)) and f.startswith('Image_src')]

            if subfolders:
                selected_subfolder = ''
                with col_folder:
                    selected_subfolder = st.selectbox('📂 请选择一个图像文件夹',subfolders, key="key_for_ImgRec_folder")
                folder_path = os.path.join(base_folder_path, selected_subfolder)
                if os.path.exists(folder_path):
                    # 加载table.csv
                    table_path = os.path.join(folder_path, "label.csv")
                    table_data = None
                    if os.path.exists(table_path):
                        #table_data = pd.read_csv("data/plate_img/Image_src03/label.csv")  # 确保加载正确
                        table_data = pd.read_csv(table_path)  # 确保加载正确
                    #显示文件夹中图片
                    image_files = os.listdir(folder_path)
                    if image_files:
                        selected_image = ''
                        with col_file:
                            selected_image = st.selectbox("🖼️ 选择一个图像进行预览", [""] + image_files,key="key_for_preview_image_unique")
                        if selected_image:
                            # 显示文件夹中图片
                            image_files = os.listdir(folder_path)
                            if image_files:
                                image_path = os.path.join(folder_path, selected_image)
                                image = Image.open(image_path)
                                col_img, col_text = st.columns([0.5, 0.5])
                                with col_img:
                                    st.image(image, caption=os.path.basename(image_path))
                                with col_text:
                                    correct_text = ''
                                    if table_data is not None:
                                        result = table_data.loc[
                                            table_data['Filename'] == selected_image, 'Recognized Text'].values
                                        if len(result) > 0:
                                            correct_text = result[0]
                                        else:
                                            correct_text = None
                                    st.write('Correct Text: ')
                                    st.write(correct_text)
                    result_title = st.empty()
                    result_display = st.empty()
                    if st.button('🚀 Start Recognition'):
                        # 使用 st.empty() 创建一个占位符
                        placeholder = st.empty()
                        # 加载前显示信息框
                        placeholder.info('正在识别图像中的钢板编号...')
                        with st.spinner('加载中，请稍候...'):
                            if os.path.exists(folder_path):
                                progress_placeholder = st.empty()
                                # 进行识别
                                data, total_images = self.process_images_from_folder(folder_path, progress_placeholder,
                                                                                IMAGE_SAVE_DIR,table_data)

                                if total_images == 0:
                                    st.warning(f'⚠️ 文件夹 {selected_subfolder} 中未找到任何图像！')
                                elif data:
                                    ru.append_to_csv(data, CSV_FILE_PATH)
                                    df = pd.DataFrame(data)
                                    self.Rec_df = df

                                    placeholder.success(
                                        f'✅ 识别完成！结果已保存到 recognized_results.csv （文件夹：{selected_subfolder}）')
                                progress_placeholder.empty()


                            else:
                                placeholder.error(f'❌ 文件夹 {folder_path} 不存在！')
                    if self.Rec_df is not None:
                        result_title.markdown("<h5 style='text-align: left; color: black;'>📋  最新识别结果：</h5>",
                                    unsafe_allow_html=True)
                        result_display.dataframe(self.Rec_df)  # 实时显示当前处理的图像结果
                    else:
                        result_title.markdown("<h5 style='text-align: left; color: black;'>📋  最新识别结果：</h5>",
                                              unsafe_allow_html=True)
                        result_display.write('暂无数据')



        elif option == op2:
            uploaded_files = st.file_uploader('📤 上传图像文件', type=['jpg', 'png', 'bmp'],
                                              accept_multiple_files=True)
            if uploaded_files:
                result_title = st.empty()
                result_display = st.empty()
                if st.button('🚀 开始识别'):
                    # 使用 st.empty() 创建一个占位符
                    placeholder = st.empty()
                    # 加载前显示信息框
                    placeholder.info('正在识别图像中的钢板编号...')
                    with st.spinner('加载中，请稍候...'):
                        progress_placeholder = st.empty()
                        data = self.process_uploaded_images(uploaded_files, progress_placeholder, IMAGE_SAVE_DIR)
                        if data:
                            ru.append_to_csv(data, CSV_FILE_PATH)
                            df = pd.DataFrame(data)
                            st.dataframe(df)  # 实时显示当前处理的图像结果
                            placeholder.success('✅ 识别完成！结果已保存到 recognized_results.csv')
                        progress_placeholder.empty()
                if self.Rec_df is not None:
                    result_title.markdown("<h5 style='text-align: left; color: black;'>📋  最新识别结果：</h5>",
                                          unsafe_allow_html=True)
                    result_display.dataframe(self.Rec_df)  # 实时显示当前处理的图像结果
                else:
                    result_title.markdown("<h5 style='text-align: left; color: black;'>📋  最新识别结果：</h5>",
                                          unsafe_allow_html=True)
                    result_display.write('暂无数据')


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


        st.header("🎥 视频编码识别")
        st.write("请选择视频输入方式 📁")


        option = st.selectbox('📥 请选择输入方式', ['项目文件夹中的视频'], key="key_for_VidRec_kinds")
        col_file,col_frame = st.columns([0.7, 0.3])


        if option == '项目文件夹中的视频':
            video_folder = 'data/plate_video'
            videos = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]

            if videos:
                select_video = ''
                frame_interval = 1
                with col_file:
                    selected_video = st.selectbox('🎬 请选择视频文件', videos, key="key_for_VidRec_file")
                with col_frame:
                    # 确定保存间隔
                    frame_interval = st.number_input("⏳ 识别帧数", min_value=1, value=20, step=1)
                # 使用 st.empty() 创建一个占位符
                placeholder = st.empty()
                result_title = st.empty()
                result_display = st.empty()
                if st.button("🚀 开始识别"):
                    # 加载前显示信息框
                    placeholder.info('正在识别视频中的钢板编号...')
                    with st.spinner('加载中，请稍候...'):
                        ru.ensure_directory_exists(video_folder)
                        video_path = os.path.join(video_folder, selected_video)
                        self.extract_unique_frames_from_video(frame_interval, video_path, frames_cache_folder,
                                                              final_frames_folder)

                        progress_placeholder = st.empty()
                        # 进行识别
                        data, total_images = self.process_images_from_folder(final_frames_folder, progress_placeholder,
                                                                             IMAGE_SAVE_DIR)

                        if total_images == 0:
                            placeholder.warning(f'⚠️ 未找到任何图像！')
                        elif data:
                            ru.append_to_csv(data, CSV_FILE_PATH)
                            df = pd.DataFrame(data)
                            placeholder.success(
                                f'✅ 识别完成！结果已保存到 recognized_results.csv')
                        progress_placeholder.empty()
                if self.Rec_df is not None:
                    result_title.markdown("<h5 style='text-align: left; color: black;'>📋  最新识别结果：</h5>",
                                          unsafe_allow_html=True)
                    result_display.dataframe(self.Rec_df)  # 实时显示当前处理的图像结果
                else:
                    result_title.markdown("<h5 style='text-align: left; color: black;'>📋  最新识别结果：</h5>",
                                          unsafe_allow_html=True)
                    result_display.write('暂无数据')
            else:
                st.write("❌ 项目文件夹中没有找到视频文件。")

            # # 显示识别结果csv表格
            # csv_display(CSV_FILE_PATH)
            #
            # # 侧边栏显示历史识别图片
            # Rec_history_image(IMAGE_SAVE_DIR)


# 创建全局实例
# img_rec_instance = ImgRec()


def csv_display(CSV_FILE_PATH):
    # 添加标题
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'>📄 结果 CSV 文件内容：</h5>", unsafe_allow_html=True)

    # 自定义 CSS，设置表格缩放
    st.markdown("""
        <style>
        .scaled-table {
            transform: scale(0.8); /* 按比例缩放表格 */
            transform-origin: top left; /* 缩放原点 */
        }
        </style>
        """, unsafe_allow_html=True)

    # 创建两个列
    col_download, col_clear = st.columns([0.5, 0.5])

    # 处理下载 CSV 的逻辑

    # 处理清除 CSV 内容的逻辑
    with col_clear:

        # 显示识别结果（CSV 表格）
        if os.path.exists(CSV_FILE_PATH):
            if ru.is_csv_empty(CSV_FILE_PATH):  # 检查 CSV 是否为空
                st.warning('⚠️ 没有可用的识别数据')
            else:
                # 清除识别结果（CSV 表格）
                if st.button('🗑️ 清除 CSV 文件内容'):
                    with st.spinner('正在清除 CSV 文件内容...'):
                        try:
                            ru.clear_csv(CSV_FILE_PATH)  # 调用自定义的清除 CSV 文件内容的函数
                            st.success('✅ CSV 文件内容已清除')
                        except Exception as e:
                            st.error(f"❌ 清除 CSV 文件时出错: {e}")
                if not ru.is_csv_empty(CSV_FILE_PATH):
                    df = pd.read_csv(CSV_FILE_PATH)
                    # 使用缩小比例显示DataFrame
                    st.markdown('<div class="scaled-table">', unsafe_allow_html=True)
                    st.dataframe(df)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning('⚠️ 没有可用的识别数据')
        else:
            st.warning('⚠️ CSV 文件不存在。')

    with col_download:
        # 读取项目中的CSV文件
        if os.path.exists(CSV_FILE_PATH):
            if ru.is_csv_empty(CSV_FILE_PATH):  # 检查 CSV 是否为空
                st.warning('⚠️ 没有可用的识别数据')
            else:
                # 读取CSV文件
                df = pd.read_csv(CSV_FILE_PATH)
                # 检查是否有“Recognized Text”和“Filename”列
                if "Recognized Text" in df.columns and "Filename" in df.columns and "Entry Time" in df.columns and "Delivery Time" in df.columns and "Batch" in df.columns:
                    # 假设 ru.generate_csv_from_column 是你自定义的函数，用来生成新的CSV文件
                    result_df = ru.generate_csv_from_column(df, "Recognized Text")
                    result_file_path = 'result/ImageRecognition_CSV/Output_steel_data.csv'
                    # 将结果保存到指定文件夹
                    result_df.to_csv(result_file_path, index=False)
                    # 下载按钮，导出结果为 CSV 文件
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="下载处理后的CSV文件",
                        data=csv,
                        file_name='Output_steel_data.csv',
                        mime='text/csv',
                    )

                    # 使用缩小比例显示处理后的结果DataFrame
                    st.markdown('<div class="scaled-table">', unsafe_allow_html=True)
                    st.dataframe(result_df)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    # 错误提示，如果必要的列不存在
                    if "Recognized Text" not in df.columns:
                        st.error("CSV文件中没有找到 'Recognized Text' 列")
                    if "Filename" not in df.columns:
                        st.error("CSV文件中没有找到 'Filename' 列")
        else:
            st.warning("⚠️ CSV 文件不存在。")

def Rec_history_image(IMAGE_SAVE_DIR):
    # 添加标题
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'>🖼️ 历史识别图片记录：</h5>", unsafe_allow_html=True)

    # 清除历史识别图片
    if st.button('🗑️ 清除图片历史'):
        with st.spinner('正在清除图片历史...'):
            ru.clear_folder(IMAGE_SAVE_DIR)
            time.sleep(0.5)  # 增加 0.5 秒的延迟
            st.success("✅ 图片历史已清除")

    # 显示历史识别的图片
    image_files = os.listdir(IMAGE_SAVE_DIR)
    if image_files:
        selected_image = st.selectbox("📂 选择一个图片进行预览", image_files, key="key_for_history_image")
        image_path = os.path.join(IMAGE_SAVE_DIR, selected_image)
        image = Image.open(image_path)
        # 在侧边栏中显示图片
        st.image(image, caption=os.path.basename(image_path), use_column_width=True)
    else:
        st.warning('⚠️ 没有可用的识别图片历史')



def plot_confidences_from_csv(csv_file):
    # 添加标题
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'>📊 历史汇总： </h5>", unsafe_allow_html=True)

    # 检查 CSV 文件是否存在
    if not os.path.exists(csv_file):
        st.error("❌ 错误：CSV 文件不存在。")
        return

    if ru.is_csv_empty(csv_file):
        st.warning("⚠️ CSV 文件为空。")
        return

    # 读取 CSV 文件
    try:
        data = pd.read_csv(csv_file)

    except Exception as e:
        st.error(f"❌ 读取 CSV 文件时出错: {e}")
        return


    # 检查是否有所需的列
    required_columns = ['Filename', 'Average Confidence', 'Timestamp', 'Accuracy']
    if not all(column in data.columns for column in required_columns):
        st.error("❌ 错误：CSV 文件缺少 'Filename', 'Average Confidence' 或 'Timestamp' 列。")
        return

    # 合并 Filename 和 Timestamp 作为 x 轴标签
    x_labels = data['Filename'] + ' ' + data['Timestamp']
    average_confidences = data['Average Confidence'].tolist()
    # 转换 'Accuracy' 列为小数
    accuracy = data['Accuracy'].str.rstrip('%').astype('float') / 100
    # 设置保存图表的目录
    save_dir = "result/Historical_barChart"
    ru.ensure_directory_exists(save_dir)

    # 设置固定颜色
    bar_color = 'rgb(0, 104, 201)'

    # 创建柱状图
    fig1 = go.Figure(data=[
        go.Bar(x=x_labels, y=average_confidences, marker_color=bar_color)
    ])
    fig1.update_layout(
        title="📈 每个文件及时间戳的平均置信度",
        xaxis_title="文件名 + 时间戳",
        yaxis_title="平均置信度",
        hoverlabel=dict(
            bgcolor="white",
            font_color="black"
        ),
        xaxis_tickangle=-45  # 将 x 轴标签旋转以防止重叠
    )

    # 获取柱子的宽度，根据元素数量动态调整
    def get_bar_width(num_positions):
        if num_positions <= 3:
            return 0.3
        elif num_positions <= 6:
            return 0.2
        else:
            return 0.1

    bar_width = get_bar_width(len(x_labels))

    # 创建组合图
    fig2 = go.Figure()

    # 添加柱状图
    fig2.add_trace(go.Bar(
        x=x_labels,
        y=accuracy,
        name='柱状图',
        width=[bar_width] * len(x_labels),
        marker_color='lightblue',  # 使用与之前相同的颜色
        hovertemplate='%{x}, %{y:.2%}',
    ))

    # 添加折线图
    fig2.add_trace(go.Scatter(
        x=x_labels,
        y=accuracy,
        mode='lines+markers',
        name='折线图'
    ))

    # 设置图表布局
    fig2.update_layout(
        title="📈 每个文件及时间戳的准确度率 - 组合图",
        xaxis_title="文件名 + 时间戳",
        yaxis_title="准确率",
        hoverlabel=dict(
            bgcolor="white",
            font_color="black"
        ),
        xaxis_tickangle=-45,  # 将 x 轴标签旋转以防止重叠
        yaxis=dict(tickformat=".2%")  # 将 y 轴刻度格式化为百分比
    )

    # 显示图表
    st.plotly_chart(fig2, use_container_width=True)
    # 显示图表
    st.plotly_chart(fig1)




def display_chart():
    # 设置保存图表的目录
    save_dir = "result/Historical_barChart"
    # 如果文件夹不存在，创建文件夹
    ru.ensure_directory_exists(save_dir)

    # 如果已经有历史图表，显示下拉框选择
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: left; color: black;'>📊 历史柱状图：</h5>", unsafe_allow_html=True)

    # 清除历史柱状图按钮
    if st.button('🗑️ 清除柱状图历史'):
        with st.spinner('正在清除柱状图历史...'):
            ru.clear_folder(save_dir)
            time.sleep(0.5)  # 增加 0.5 秒的延迟
            st.success("✅ 柱状图历史已清除")

    # 加载历史柱状图
    historical_charts = ru.load_historical_charts(save_dir)
    if historical_charts:
        selected_chart = st.selectbox("📂 选择查看的历史柱状图", list(historical_charts.keys()))

        # 根据选择的图表显示相应的图表
        if selected_chart:
            chart_file_path = historical_charts[selected_chart]
            # 使用 st.components.v1.html 来显示保存的 HTML 图表
            with open(chart_file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=600)
    else:
        st.warning("⚠️ 当前没有可用的历史柱状图。")

# 绘制并保存柱状图
def plot_confidences(average_confidences):
    # 设置保存图表的目录
    save_dir = "result/Historical_barChart"
    # 如果文件夹不存在，创建文件夹
    ru.ensure_directory_exists(save_dir)
    # 加载历史柱状图
    historical_charts = ru.load_historical_charts(save_dir)

    chart_name = f"Chart {len(historical_charts) + 1}"
    # 设置固定颜色
    bar_color = 'rgb(0, 104, 201)'  # 你可以自定义任何 RGB 或 HEX 颜色，如 rgb(0, 104, 201)

    fig = go.Figure(data=[
        go.Bar(x=list(range(len(average_confidences))), y=average_confidences, marker_color=bar_color)
    ])
    fig.update_layout(
        title="📈 每次 OCR 调用的平均置信度",
        xaxis_title="OCR 调用编号",
        yaxis_title="平均置信度",
        # 设置 hover 信息的背景颜色为白色，字体颜色为黑色
        hoverlabel=dict(
            bgcolor="white",
            font_color="black"
        )
    )
    # 保存柱状图到文件
    ru.save_chart_to_file(fig, chart_name, save_dir)
    st.plotly_chart(fig)


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


def calculate_accuracy(recognized_text, correct_text):
    # 去掉识别文本和正确文本中的空格用于匹配
    recognized_text = recognized_text.replace(" ", "")
    correct_text = correct_text.replace(" ", "").replace("\n", "")

    # 获取最小的长度，防止索引越界
    min_len = min(len(recognized_text), len(correct_text))

    # 统计匹配的字符个数
    match_count = sum(1 for i in range(min_len) if recognized_text[i] == correct_text[i])

    # 准确率 = 匹配字符数 / 正确编码的总长度
    accuracy = match_count / len(correct_text) if len(correct_text) > 0 else 0
    return accuracy





