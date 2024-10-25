import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
from langchain.prompts import PromptTemplate
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from datetime import datetime

# Configuration for image classification model
class_names = ['Calculus', 'Dental Caries', 'Gingivitis', 'Hypodontia', 'Tooth Discoloration']
model = load_model('data/Model3.keras')

def classify_images(image):
    # Check if the image is None
    if image is None:
        return "No image uploaded. Please upload a dental image."

    # Resize and preprocess the image
    try:
        input_image = tf.image.resize(image, (180, 180))  # Resize to expected input size
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array, axis=0)

        # Make predictions
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        
        # Prepare the outcome message
        outcome = f'The image belongs to {class_names[np.argmax(result)]} with a score of {np.max(result) * 100:.2f}%'
        return outcome
    except Exception as e:
        return f"Error processing the image: {str(e)}"

    
# Configuration for the medical chatbot
DB_FAISS_PATH = "vectorstores/db_faiss"
custom_prompt_template = """
Use the following pieces of information
to answer the user's question. If you don't know the answer, 
please just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    return PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])

def load_llm():
    return CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.7
    )

def retrieval_qa_chain(llm, prompt, db):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

# Charger le modèle et la base de données une seule fois
def create_qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    return retrieval_qa_chain(llm, qa_prompt, db)

# Charger le modèle une seule fois au démarrage
qa_bot_instance = create_qa_bot()
history = []
question_counter = 1  # Initialisation du compteur

def answer_query(query):
    global question_counter  # Utiliser le compteur global
    if history and history[-1].startswith(f"**Question:** {query}"):
        return history[-1]

    response = qa_bot_instance.invoke({'query': query})
    answer = response['result']
    sources = response['source_documents']

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    final_answer = answer
    
    if sources:
        formatted_sources = "\n\n---\n**Sources:**\n" + "\n".join([str(source) for source in sources])
    else:
        formatted_sources = "\n\n---\n**Sources:** No sources found."

    # Ajout de la question numérotée à l'historique
    history.append(f"{current_time}\n**Question {question_counter}:** {query}\n**Answer:** {final_answer}\n{formatted_sources}\n\n")
    question_counter += 1  # Incrémentation du compteur

    return final_answer, "\n".join(history)

# Create the main Gradio interface
with gr.Blocks(css=".gradio-container {max-width: 400px; margin: auto;}") as app:
    gr.Markdown("<h1 style='text-align: center;'>Dental and Medical Assistant</h1>")
        
    with gr.Tab("Image Classification"):
        with gr.Row():
            image_input = gr.Image(type='numpy', label="Upload Dental Image")
            classification_output = gr.Label(num_top_classes=5, label="Classification Results")
        image_input.change(fn=classify_images, inputs=image_input, outputs=classification_output)
    
    with gr.Tab("Medical Chatbot"):
        with gr.Row():
            query_input = gr.Textbox(label="Ask a medical question")
            chatbot_output = gr.Textbox(label="Answer", lines=5)
        with gr.Row():
            history_output = gr.Textbox(label="Chat History", value="", lines=8, interactive=False)
        query_input.submit(fn=answer_query, inputs=query_input, outputs=[chatbot_output, history_output])

app.launch(share=True, show_api=False)
