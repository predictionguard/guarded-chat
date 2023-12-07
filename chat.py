import time

import predictionguard as pg
from langchain import PromptTemplate, FewShotPromptTemplate
import streamlit as st

#--------------------------#
# Prompt templates         #
#--------------------------#

demo_formatter_template = """\nUser: {user}
Assistant: {assistant}\n"""
demo_prompt = PromptTemplate(
    input_variables=["user", "assistant"],
    template=demo_formatter_template,
)


#---------------------#
# Streamlit config    #
#---------------------#

#st.set_page_config(layout="wide")

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#--------------------------#
# Streamlit sidebar        #
#--------------------------#

st.sidebar.title("Chat Playground")
st.sidebar.markdown(
    "This is a playground for [Prediction Guard](https://www.predictionguard.com) based chat assistants. "
    "You can try out different models and configurations and see how the assistant responds."
)

st.sidebar.markdown("## Model Configuration")
model = st.sidebar.selectbox(label="Model", options=["Yi-34B"])
# temperature = st.sidebar.slider(
#     label="Temperature",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.75,
#     step=0.01,
#     format="%f",
# )
# max_tokens = st.sidebar.slider(
#     label="Max tokens",
#     min_value=1,
#     max_value=1000,
#     value=200,
#     step=10,
#     format="%d",
# )
# top_p = st.sidebar.slider(
#     label="Top P",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.75,
#     step=0.01,
#     format="%f",
# )
# top_k = st.sidebar.slider(
#     label="Top K",
#     min_value=0.0,
#     max_value=1.0,
#     value=0.75,
#     step-0.01,
#     format="%f",
# )

st.sidebar.markdown("## Guards/checks")
consistency = st.sidebar.checkbox("Consistency", value=False)
factuality = st.sidebar.checkbox("Factuality", value=False)
toxicity = st.sidebar.checkbox("Toxicity", value=False)


#--------------------------#
# Streamlit app            #
#--------------------------#

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # contruct prompt
        examples = []
        turn = "user"
        example = {}
        for m in st.session_state.messages:
            latest_message = m["content"]
            example[turn] = m["content"]
            if turn == "user":
                turn = "assistant"
            else:
                turn = "user"
                examples.append(example)
                example = {}
        if len(example) > 4:
            examples = examples[-4:]

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=demo_prompt,
            example_separator="",
            prefix="The following is a conversation between an AI assistant and a human user. The assistant is helpful, creative, clever, and very friendly.\n",
            suffix="\nHuman: {human}\nAssistant: ",
            input_variables=["human"],
        )

        content = few_shot_prompt.format(human=latest_message)

        # generate response
        with st.spinner("Thinking..."):
            result = pg.Chat.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                # max_tokens=max_tokens,
                # temperature=temperature,
                # top_p=top_p,
                # top_k=top_k,
                # output = {
                #     "consistency": consistency,
                #     "factuality": factuality,
                #     "toxicity": toxicity
                # }
            )

        completion = result['choices'][0]['message']['content']
        completion = completion.split("Human:")[0].strip()
        completion = completion.split("H:")[0].strip()
        completion = completion.split('#')[0].strip()
        for token in completion.split(" "):
            full_response += " " + token
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.075)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})