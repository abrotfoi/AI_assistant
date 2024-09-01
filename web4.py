import streamlit as st
from langchain.memory import ConversationBufferMemory
from utils3 import qa_agent
from utils2 import get_chat_response

def main():
    st.title("📑 AI知识库小助手")

    with st.sidebar:
        openai_api_key = st.text_input("请输入dashscope api密钥：", type="password")
        base_url = st.text_input("base_url：", type="password")
        st.markdown("[获取dashscope api key](https://account.aliyun.com/)")

    selected_method = st.radio(
        "你想选择哪种模式进行对话？",
        ["None", "chat_qa_chain"],
        captions=["不使用检索问答的普通模式", "带历史记录的检索问答模式"])
    if selected_method == "None":
        if "memory" not in st.session_state:
            st.session_state["memory"] = ConversationBufferMemory(return_messages=True)
            st.session_state["messages"] = [{"role": "ai",
                                             "content": "你好，我是你的AI助手，有什么可以帮你的吗？"}]

        for message in st.session_state["messages"]:
            st.chat_message(message["role"]).write(message["content"])

        prompt = st.chat_input()
        if prompt:
            if not openai_api_key:
                st.info("请输入你的OpenAI API Key")
                st.stop()
            st.session_state["messages"].append({"role": "human", "content": prompt})
            st.chat_message("human").write(prompt)

            with st.spinner("AI正在思考中，请稍等..."):
                response = get_chat_response(prompt, st.session_state["memory"],
                                             openai_api_key, base_url)
            msg = {"role": "ai", "content": response}
            st.session_state["messages"].append(msg)
            st.chat_message("ai").write(response)


    elif selected_method == "chat_qa_chain":
        if "memory1" not in st.session_state:
            st.session_state["memory1"] = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                output_key="answer"
            )

        uploaded_file = st.file_uploader("上传你的PDF文件：", type="pdf")
        question = st.text_input("对PDF的内容进行提问", disabled=not uploaded_file)

        if uploaded_file and question and not openai_api_key:
            st.info("请输入你的OpenAI API密钥")

        if uploaded_file and question and openai_api_key:
            with st.spinner("AI正在思考中，请稍等..."):
                response = qa_agent(openai_api_key, st.session_state["memory1"],
                                    uploaded_file, question, base_url)
            st.write("### 答案")
            st.write(response["answer"])
            st.session_state["chat_history"] = response["chat_history"]

        if "chat_history" in st.session_state:
            with st.expander("历史消息"):
                for i in range(0, len(st.session_state["chat_history"]), 2):
                    human_message = st.session_state["chat_history"][i]
                    ai_message = st.session_state["chat_history"][i + 1]
                    st.write(human_message.content)
                    st.write(ai_message.content)
                    if i < len(st.session_state["chat_history"]) - 2:
                        st.divider()

if __name__ == "__main__":
    main()