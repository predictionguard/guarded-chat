# Guarded Chatbot

Create a custom domain chatbot with checks for factuality, toxicity, and consistency.

## Installation and Setup

- Install the package requirements by running `pip install -r requirements.txt`
- Get a Prediction Guard access token (as described [here](https://docs.predictionguard.com/)) and set it as the environment variable `PREDICTIONGUARD_TOKEN`.

## How the Chatbot Works
- The chatbot UI is created through [Streamlit](https://streamlit.io), which is used to create various UI elements in Python.
- When a prompt is entered, it is sent to the Prediction Guard API through the Python client, using the Prediction Guard chat function. The code for the call is:
```python
result = pg.Chat.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                output = {
                    "consistency": consistency,
                    "factuality": factuality,
                    "toxicity": toxicity
                }
            )
```
- This function takes in the model settings along with the message, sends it to the API, and allows for the chat message to be processed by the LLMs.

## Running the Chatbot

- To run the chatbot, first define the `PREDICTIONGUARD_TOKEN` environment variable, then run `streamlit run chat.py`.
- The chatbot will automatically open in your browser, but if it does not, the terminal will output the URL for the app.
- Any errors that the chatbot has will print out in both the app and the terminal, so you can monitor an issues more easily.

## Using the Chatbot

- The chatbot allows for you to make calls to the Prediction Guard Chat endpoint, in a the visual format of a chatbot. 
- In the sidebar, you can configure the settings of the chatbot, such as the model used and whether to have consistency, factuality, and toxicity checks.
- You can also customize how you want the model to handle the message, such as modifying the length of the response by changing the amount of tokens, changing the temperature for more or less consistent responses, or the top_p and top_k to handle how the model chooses which tokens to use.

## Mandarin Chatbot
- The Prediction Guard chatbot also has a Mandarin version, which utilizes the Yi-34B-Chat LLM. 
- This model functions the same way as the English version, but is also able to process Mandarin.
- To run it, use the `chat-mandarin.py` program instead of `chat.py`