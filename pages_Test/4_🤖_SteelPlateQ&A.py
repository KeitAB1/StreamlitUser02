import streamlit as st
import requests
from special_questions import handle_special_questions  # 引入特殊问题处理函数

# 从 secrets.toml 文件中获取信息
openweboi_url = st.secrets.get("openweboi_url")
email = st.secrets.get("email")
streamlit_password = st.secrets.get("streamlit_password")  # 用于访问 Streamlit 应用的密码
openwebui_password = st.secrets.get("openwebui_password")  # 用于登录 OpenWebUi 的密码

# openweboi_url = ""
# email = ""
# streamlit_password = ""
# openwebui_password = ""


# 定义API类
class FreeApi:
    def __init__(self, keyword):
        self.api_url = f'http://api.qingyunke.com/api.php?key=free&appid=0&msg={keyword}'

    def get_result(self):
        try:
            response = requests.get(self.api_url)
            if response.status_code == 200:
                try:
                    return response.json()  # 解析为JSON
                except ValueError:
                    return {"error": "Response is not valid JSON.", "content": response.text}
            else:
                return {"error": "Failed to fetch data from the API."}
        except Exception as e:
            return {"error": str(e)}


# 初始化对话历史的 session_state
if 'dialog_history' not in st.session_state:
    st.session_state['dialog_history'] = []

# 页面标题
st.title("Steel Plate Intelligent Q&A System")

# 显示英文提示信息
st.write(
    "The Steel Plate Intelligent Q&A feature is currently being updated. However, you can access it through OpenWebUi via internal network tunneling.")

# 添加一个按钮，点击后跳转到 OpenWebUi 的链接
if st.button("Enter via OpenWebUi"):
    if openweboi_url:
        st.write("Redirecting to OpenWebUi...")
        st.markdown(f"[Click here to access OpenWebUi]({openweboi_url})")
    else:
        st.error("OpenWebUi URL not found. Please check your secrets.toml file.")

st.write("#### Simple version assistant")
st.write("(functional testing is being completed...)")

# 侧栏创建AI参数设置
with st.sidebar:


    # 密码输入框，用于 Streamlit 应用的访问
    entered_streamlit_password = st.text_input("Enter Streamlit Password", type="password")
    if st.button("Submit Streamlit Password"):
        if entered_streamlit_password == streamlit_password:
            st.success("Access granted! Please enter the OpenWebUi password to view credentials.")

            # 输入 OpenWebUi 密码
            entered_openwebui_password = st.text_input("Enter OpenWebUi Password", type="password")
            if st.button("Submit OpenWebUi Password"):
                if entered_openwebui_password == openwebui_password:
                    st.success("Correct! Here are the credentials to OpenWebUi:")
                    st.write(f"**Email:** {email}")  # 使用从 secrets.toml 中读取的邮箱
                    st.write(f"**Password:** {openwebui_password}")  # 使用从 secrets.toml 中读取的 OpenWebUi 密码
                else:
                    st.error("Incorrect OpenWebUi password! Please try again.")
        else:
            st.error("Incorrect Streamlit password! Please try again.")

    st.header("AI Chat Settings")
    temperature = st.slider("Response Temperature", 0.0, 1.0, 0.7)
    max_length = st.slider("Max Response Length", 10, 100, 50)
    st.write("Adjust these settings to control the behavior of the AI responses.")


# 回车键触发函数
def handle_user_input():
    user_question = st.session_state.user_input
    if user_question:
        # 处理特殊问题
        special_answer = handle_special_questions(user_question)

        if special_answer:
            st.session_state['dialog_history'].append({"role": "user", "content": user_question})
            st.session_state['dialog_history'].append({"role": "bot", "content": special_answer})
        else:
            # 调用API获取机器人回复
            api = FreeApi(user_question)
            result = api.get_result()

            if "content" in result:
                response = result['content']
                st.session_state['dialog_history'].append({"role": "user", "content": user_question})
                st.session_state['dialog_history'].append({"role": "bot", "content": response})
            elif "error" in result:
                st.session_state['dialog_history'].append({"role": "bot", "content": f"Error: {result['error']}"})

        # 清空输入框
        st.session_state.user_input = ''


# 输入框，回车键触发
st.text_input("Enter your question here:", key="user_input", on_change=handle_user_input)

# 显示对话历史，使用 Emoji 并左右对齐
if st.session_state['dialog_history']:
    for chat in st.session_state['dialog_history']:
        if chat["role"] == "user":
            # 用户消息靠左显示
            st.markdown(
                f"""
                <div style='text-align: left; background-color: #daf6e2; padding: 10px; border-radius: 10px; margin: 10px 0;'>
                🙂 <strong>User</strong>: {chat['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            # 机器人消息靠右显示
            st.markdown(
                f"""
                <div style='text-align: right; background-color: #f0f0f5; padding: 10px; border-radius: 10px; margin: 10px 0;'>
                🤖 <strong>Bot</strong>: {chat['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
