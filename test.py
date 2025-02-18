from langchain_community.llms.gpt4all import GPT4All
import os

# Asegurarnos de que la ruta es absoluta
model_path = os.path.abspath("./models/llama-2-7b-chat.ggmlv3.q4_0.bin")

# Inicializar el modelo con configuración específica
llm = GPT4All(
    model=model_path,
    backend='llama',
    verbose=True,
    n_threads=8,  # Ajusta esto según los núcleos de tu CPU
    max_tokens=2048,
    temp=0.7,
    top_p=0.95,
    repeat_penalty=1.1
)

# Probar con un prompt estructurado
prompt = """[INST] <<SYS>>
You are a helpful AI assistant.
<</SYS>>
¿Cuál es la capital de Peru? [/INST]"""

try:
    response = llm(prompt)
    print("Respuesta:", response)
except Exception as e:
    print(f"Error: {str(e)}")
    print(f"Tipo de error: {type(e)}")