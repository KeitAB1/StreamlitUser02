import streamlit as st
import easyocr
from auxiliary import ImgRec_fun as ir
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 加载模型
@st.cache_resource
def load_ocr_model():
    # 使用 st.spinner 来显示加载提示
    with st.spinner("模型加载中......"):
        reader = easyocr.Reader(['en'], model_storage_directory="data/easyorc_models")
        imageRec = ir.ImgRec()
        imageRec.set_reader(reader)
    return reader,imageRec

# 加载 OCR 模型
reader, imageRec  = load_ocr_model()

# 使用 Img_Rec.py 中的全局实例设置模型
# ir.img_rec_instance.set_reader(reader)

# 定义图片和CSV文件保存路径
IMAGE_SAVE_DIR = 'result/ImageRecognition_Img'
CSV_FILE_DIR = 'result/ImageRecognition_CSV'
CSV_FILE_PATH = CSV_FILE_DIR + '/recognized_results.csv'
VIDEO_DIR = 'data/plate_video'  # 保存视频的目录


def main():
    """主函数"""
    st.markdown("<h1 style='text-align: center; color: black;'>🔍钢 板 编 码 识 别</h1>",
                unsafe_allow_html=True)
    st.markdown("<br><br>", unsafe_allow_html=True)
    # 使用 display_icon_with_header 函数替换部分的展示
    col1, col2, col11 = st.columns([0.01, 0.25, 0.55])
    with col1:
        st.image("data/icon/icon02.jpg", width=20)
    with col2:
        # 选择识别模式：图像识别或视频识别
        mode = st.selectbox('请选择识别模式（📷/🎥）', ['图像识别 📷', '视频识别 🎥'])





    if mode == '图像识别 📷':
        #ir.img_rec_instance.Image_Recongnotion(IMAGE_SAVE_DIR, CSV_FILE_PATH)
        imageRec.Image_Recongnotion(IMAGE_SAVE_DIR, CSV_FILE_PATH)
    elif mode == '视频识别 🎥':
        #ir.img_rec_instance.Video_Recognition(IMAGE_SAVE_DIR,CSV_FILE_PATH)
        imageRec.Video_Recognition(IMAGE_SAVE_DIR,CSV_FILE_PATH)


    # 在侧边栏添加一个复选框
    toggle_state = st.sidebar.checkbox("显示详细数据")  # 复选框，类似开关
    # 根据复选框状态显示不同的内容
    if toggle_state:
        #ir.Rec_history_image(IMAGE_SAVE_DIR)
        ir.csv_display(CSV_FILE_PATH)
        ir.plot_confidences_from_csv(CSV_FILE_PATH)
        #ir.display_chart()
    else:
        st.write('详细数据已隐藏，可在侧边栏打开')

    # ir.img_rec_instance.mask_Settings()
    # ir.Rec_history_image(IMAGE_SAVE_DIR)
    # ir.csv_display(CSV_FILE_PATH)
    # ir.plot_confidences_from_csv(CSV_FILE_PATH)
    # ir.display_chart()
    #ir.plot_confidences(ir.img_rec_instance.average_confidences)


if __name__ == "__main__":
    main()
