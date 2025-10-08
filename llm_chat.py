# Import necessary packages
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai import Credentials
from langchain_ibm import WatsonxLLM
import gradio as gr

# ✅ Use a supported model (Mixtral 8x7B is NOT available)
# You can pick one of these supported models:
# 'ibm/granite-13b-instruct-v2'
# 'ibm/granite-3-3-8b-instruct'
# 'mistralai/mistral-medium-2505'
# 'mistralai/mistral-small-3-1-24b-instruct-2503'

model_id = "ibm/granite-3-3-8b-instruct"


# Set generation parameters
parameters = {
    GenParams.MAX_NEW_TOKENS: 256,   # Maximum number of tokens to generate
    GenParams.TEMPERATURE: 0.5,      # Controls creativity / randomness
}

# Your IBM watsonx.ai project ID
project_id = "skills-network"

# 🔐 If you need credentials, ensure your environment has these set:
# export WATSONX_APIKEY=<your_api_key>
# export WATSONX_URL="https://us-south.ml.cloud.ibm.com"

# Initialize the model for inference
watsonx_llm = WatsonxLLM(
    model_id=model_id,
    url="https://us-south.ml.cloud.ibm.com",
    project_id=project_id,
    params=parameters,
)

# Define a function for generating responses
def generate_response(prompt_txt):
    generated_response = watsonx_llm.invoke(prompt_txt)
    return generated_response

# Build the Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
    allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Watsonx.ai Chatbot",
    description="Ask any question and the chatbot will try to answer using IBM watsonx.ai LLM."
)

# Launch the app locally
chat_application.launch(server_name="127.0.0.1", server_port=7860)
