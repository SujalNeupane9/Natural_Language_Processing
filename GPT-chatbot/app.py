import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    # Load the model and tokenizer
    model_name = "Neupane9Sujal/GPT2-chatbot"  
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define a function for model inference
    @st.cache(allow_output_mutation=True)
    def generate_response(user_input):
        input_ids = tokenizer.encode(user_input, return_tensors="pt")
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)
        response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
        return response

    # Streamlit app
    st.title("Conversational AI")
    user_input = st.text_input("User Input", "")

    if user_input:
        with st.spinner("Generating response..."):
            response = generate_response(user_input)
        st.text_area("Model Response", value=response, height=200)

if __name__ == "__main__":
    main()
