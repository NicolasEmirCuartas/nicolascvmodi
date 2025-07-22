from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import csv
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes.retriever import DensePassageRetriever
from haystack.nodes.reader import FARMReader
from haystack.pipelines import Pipeline as ExtractiveQAPipeline
import os
from rapidfuzz import process

app = Flask(__name__)

# 1. Cargar modelo GPT-2 una sola vez
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 2. Configuración Haystack y carga de CSV
document_store = InMemoryDocumentStore()
ruta_csv = os.path.join(os.path.dirname(__file__), "infonico.csv")

def leer_csv(ruta):
    datos = []
    try:
        with open(ruta, mode="r", encoding="utf-8") as archivo:
            lector = csv.DictReader(archivo)
            for fila in lector:
                datos.append(fila)
    except Exception as e:
        print(f"Error CSV: {e}")
    return datos

def cargar_datos_en_haystack(datos_csv):
    docs = [{"content": fila["Información"]} for fila in datos_csv if "Información" in fila]
    document_store.write_documents(docs)

retriever = DensePassageRetriever(
    document_store=document_store,
    query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
    passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
    use_gpu=False,
    batch_size=16,
    embed_title=False
)

reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
pipeline_rag = ExtractiveQAPipeline()
pipeline_rag.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipeline_rag.add_node(component=reader, name="Reader", inputs=["Retriever"])

datos_csv = leer_csv(ruta_csv)
if datos_csv:
    cargar_datos_en_haystack(datos_csv)

def buscar_con_rag(pregunta):
    prediction = pipeline_rag.run(query=pregunta, params={"Retriever": {"top_k": 3}, "Reader": {"top_k": 1}})
    if prediction["answers"]:
        return prediction["answers"][0].answer
    if prediction.get("documents"):
        return prediction["documents"][0].content
    return "No se encontró respuesta."

# Preguntas y respuestas predefinidas
respuestas_predefinidas = {
    "hola": (
        "¡Hola! ¿En qué puedo ayudarte sobre mi experiencia laboral, formación académica  o perfil profesional ?"
    ),
    "¿cómo está el día?": "¡Muy bien, gracias! Listo para responder cualquier consulta sobre mi perfil profesional.",
    "¿cómo te sentís hoy?": "Me siento motivado y con muchas ganas de afrontar nuevos desafíos.",
    "¿cómo estás?": "¡Muy bien! Gracias por preguntar.",
    "¿cuál es tu nombre?": "Mi nombre completo es Nicolás Emir Cuartas.",
    "¿qué edad tienes?": "Tengo 28 años.",
    "cuantos años tienes": "Tengo 28 años.",
    "¿dónde naciste?": "Nací en La Plata, Buenos Aires, Argentina.",
    "¿qué estudios tienes?": "Licenciatura en Sistemas y Ciencia de Datos (no finalizadas).",
    "¿qué cursos has realizado?": "Universidad de Helsinki, Harvard, Microsoft: IA con Python.",
    "¿qué certificado tienes?": "Python - Universidad del Plata.",
    "¿qué experiencia laboral tienes?": "Fundación: Gestión de datos de mercancías y ventas durante la pandemia.",
    "¿qué proyectos tecnológicos hiciste?": "Página web para un gimnasio (HTML, CSS, JavaScript), Aplicación para el club Universal de La Plata (Data Analyst para socios).",
    "¿cuáles son tus hobbies?": "Basket: Técnico a cargo de un grupo de personas de 20 a 23 años.",
    "¿qué habilidades tienes?": "Adaptación al grupo, Manejo de la presión, Creatividad.",
    "resumen": (
        "Soy un chico que intenta involucrarse en el mundo de las IA. Soy carismático, lo que me permite adaptarme perfectamente a los grupos. "
        "Estudiante de Licenciatura en Sistemas y Ciencia de Datos, me apasiona la programación. Comencé con un curso de Python en la Universidad del Plata y continué con cursos en Harvard, la Universidad de Helsinki y Microsoft sobre IA con Python. "
        "Durante la pandemia, gestioné datos de mercancías y ventas en la Cueva Solidaria. Desarrollé una página web para un gimnasio utilizando HTML, CSS y JavaScript. "
        "Fui entrenador de básquet para jóvenes de 20 a 23 años, liderando y gestionando datos relacionados. También trabajé en una financiera manejando Excel y en una empresa mejorando datos para su aplicación."
    ),
    "¿me podés pasar tu cv?": "¡Por supuesto! Podés ver y descargar mi CV actualizado en el siguiente enlace: https://nicolasemircuartas.github.io/cvnicolas/",
    "¿dónde puedo ver tu currículum?": "Mi currículum está disponible online aquí: https://nicolasemircuartas.github.io/cvnicolas/",
    "¿tienes un enlace a tu cv?": "Sí, podés acceder a mi CV en este link: https://nicolasemircuartas.github.io/cvnicolas/",
    "¿qué habilidades debo destacar para este puesto?": "Para destacar en este puesto, deberías resaltar tus habilidades técnicas relacionadas con el rol, como el manejo de herramientas específicas o lenguajes de programación, así como tus habilidades blandas, como la comunicación efectiva, el trabajo en equipo y la resolución de problemas. Además, menciona cualquier experiencia previa relevante que demuestre tu capacidad para cumplir con las responsabilidades del puesto.",
    "¿por qué te interesa el área de ciencia de datos?": "Me interesa la ciencia de datos porque combina mi pasión por la programación y el análisis de información para resolver problemas reales. Disfruto descubrir patrones en los datos y aportar valor a través de soluciones basadas en evidencia.",
    "¿cómo aplicaste tus conocimientos de programación en proyectos reales?": "Desarrollé una página web para un gimnasio utilizando HTML, CSS y JavaScript, y creé una aplicación para el club Universal de La Plata donde analicé datos de socios. Estos proyectos me permitieron aplicar mis habilidades técnicas y aprender sobre gestión de proyectos.",
    "¿qué experiencia tienes liderando equipos?": "Fui entrenador de básquet para jóvenes de 20 a 23 años, donde desarrollé habilidades de liderazgo, comunicación y gestión de grupos. Esta experiencia me enseñó a motivar y coordinar equipos para alcanzar objetivos comunes.",
    "¿cómo manejas la presión y los desafíos?": "Manejo la presión priorizando tareas, manteniendo la calma y buscando soluciones creativas. Como entrenador y en mi experiencia laboral, aprendí a adaptarme rápidamente y a tomar decisiones efectivas bajo situaciones exigentes.",
    "¿qué logros destacarías de tu carrera hasta ahora?": "Uno de mis logros fue gestionar datos de mercancías y ventas durante la pandemia en una fundación, optimizando procesos en un contexto desafiante. También fui reconocido como mejor entrenador del club en 2022 por mi liderazgo y compromiso.",
    "¿qué habilidades técnicas y blandas consideras tus fortalezas?": "Mis fortalezas técnicas incluyen Python, análisis de datos, HTML, CSS, JavaScript y manejo de herramientas como Excel y Power BI. En cuanto a habilidades blandas, destaco la adaptabilidad, la creatividad, el trabajo en equipo y la comunicación efectiva.",
    "¿cómo te mantienes actualizado en tecnología?": "Me mantengo actualizado realizando cursos en instituciones como Harvard, la Universidad de Helsinki y Microsoft, y participando en comunidades online. Además, aplico lo aprendido en proyectos personales y colaborativos.",
    "¿cómo resolvés conflictos en un equipo?": "Escucho activamente a todas las partes, busco puntos en común y propongo soluciones que beneficien al grupo. Creo que la comunicación abierta y el respeto son claves para resolver cualquier conflicto.",
    "¿qué te motiva profesionalmente?": "Me motiva aprender constantemente, enfrentar nuevos desafíos y ver cómo mi trabajo puede generar un impacto positivo en las personas y las organizaciones.",
    "¿por qué deberíamos contratarte?": "Porque combino una sólida formación técnica con experiencia práctica y habilidades de liderazgo. Soy adaptable, proactivo y tengo muchas ganas de aportar valor y crecer junto a la empresa.",
}

respuestas_predefinidas.update({
    "perfil": (
        "Perfil Profesional: Profesional argentino en ciencia de datos, apasionado por la programación y la inteligencia artificial. "
        "Actualmente estoy estudiando y formándome en inteligencia artificial, con experiencia en análisis de datos, desarrollo web y liderazgo deportivo. "
        "Estudiante de Licenciatura en Sistemas y Ciencia de Datos. "
        "Si querés más información, podés ver mi LinkedIn: https://www.linkedin.com/in/nicolas-emir-cuartas-166878161"
    ),
    "perfil profesional": (
        "Perfil Profesional: Profesional argentino en ciencia de datos, apasionado por la programación y la inteligencia artificial. "
        "Actualmente estoy estudiando y formándome en inteligencia artificial, con experiencia en análisis de datos, desarrollo web y liderazgo deportivo. "
        "Estudiante de Licenciatura en Sistemas y Ciencia de Datos. "
        "Para más información, visitá mi LinkedIn: https://www.linkedin.com/in/nicolas-emir-cuartas-166878161"
    ),
    "nicolás": (
        "Nicolás Emir Cuartas es un profesional argentino en ciencia de datos, apasionado por la programación y la inteligencia artificial. "
        "Actualmente estudia y se forma en inteligencia artificial, con experiencia en análisis de datos, desarrollo web y liderazgo deportivo. "
        "Más info en mi LinkedIn: https://www.linkedin.com/in/nicolas-emir-cuartas-166878161"
    ),
    "quién es nicolás": (
        "Nicolás Emir Cuartas es un profesional argentino en ciencia de datos, apasionado por la programación y la inteligencia artificial. "
        "Podés ver su perfil completo en LinkedIn: https://www.linkedin.com/in/nicolas-emir-cuartas-166878161"
    ),
    "contame de nicolás": (
        "Nicolás Emir Cuartas es estudiante de Licenciatura en Sistemas y Ciencia de Datos, con experiencia en IA, análisis de datos y desarrollo web. "
        "Más información en su LinkedIn: https://www.linkedin.com/in/nicolas-emir-cuartas-166878161"
    ),
    "contame lo que sabes de nicolás": (
        "Nicolás Emir Cuartas es un profesional argentino en ciencia de datos, apasionado por la programación y la inteligencia artificial. "
        "Más info en LinkedIn: https://www.linkedin.com/in/nicolas-emir-cuartas-166878161"
    ),
    "sobre nicolás": (
        "Podés conocer más sobre Nicolás Emir Cuartas en su LinkedIn: https://www.linkedin.com/in/nicolas-emir-cuartas-166878161"
    ),
})

def generar_respuesta(pregunta, info_relevante=""):
    contexto = (
        "Nicolás Emir Cuartas, profesional argentino en ciencia de datos, "
        "con experiencia en economía, análisis de datos, desarrollo web y liderazgo deportivo. "
        "Responde de manera profesional, clara y concisa, usando la información relevante si aplica."
    )
    prompt = (
        f"{contexto}\n"
        f"Información relevante: {info_relevante}\n"
        f"Pregunta: {pregunta}\n"
        f"Respuesta profesional:"
    )
    try:
        resultado = generator(prompt, max_new_tokens=80, do_sample=True, temperature=0.7)
        respuesta = resultado[0]["generated_text"].split("Respuesta profesional")[-1].strip()
        respuesta = respuesta.replace("(máximo 3 frases):", "").strip()
        respuesta = ". ".join(respuesta.split(".")[:3]).strip() + "."
        return respuesta
    except Exception as e:
        return f"Error: {str(e)}"

def buscar_respuesta_csv(pregunta_usuario):
    datos = leer_csv(ruta_csv)
    for fila in datos:
        if "Pregunta" in fila and "Respuesta" in fila:
            if pregunta_usuario.strip().lower() == fila["Pregunta"].strip().lower():
                return fila["Respuesta"]
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/inicioia")
def inicioia():
    return render_template("inicioia.html")

def responder_seccion(seccion):
    datos = leer_csv(ruta_csv)
    for fila in datos:
        if fila.get("Sección", "").strip().lower() == seccion.strip().lower():
            info = fila.get("Información", "").strip()
            if info:
                return info
    return f"No se encontró información para la sección: {seccion}."

LINKEDIN = "https://www.linkedin.com/in/nicolas-emir-cuartas-166878161"

@app.route("/preguntar", methods=["POST"])
def preguntar():
    datos = request.get_json()
    pregunta = datos.get("pregunta", "").strip().lower()

    if "formacion" in pregunta or "formación" in pregunta or "formacion academica" in pregunta or "formación académica" in pregunta:
        return jsonify({"respuesta": (
            f"Formación académica:\n"
            f"- Carrera: {responder_seccion('Carreras en Curso')}\n"
            f"- Cursos: {responder_seccion('Cursos Realizados')}\n"
            f"- Certificado: {responder_seccion('Certificado')}\n"
        )})
    if "habilidad" in pregunta:
        return jsonify({"respuesta": f"Habilidades:\n{responder_seccion('Habilidades')}\n"})
    if "experiencia" in pregunta:
        return jsonify({"respuesta": f"Experiencia Laboral:\n{responder_seccion('Experiencia Laboral')}\n\n¿En qué más puedo ayudarte?"})

    # Busca en el CSV
    respuesta_csv = buscar_respuesta_csv(pregunta)
    if respuesta_csv:
        return jsonify({"respuesta": respuesta_csv})

    # Busca en el diccionario
    if pregunta in respuestas_predefinidas:
        return jsonify({"respuesta": respuestas_predefinidas[pregunta]})

    # Busca respuesta aproximada
    respuesta_aproximada = buscar_respuesta_aproximada(pregunta, respuestas_predefinidas)
    if respuesta_aproximada:
        return jsonify({"respuesta": respuesta_aproximada})

    # Busca con RAG o IA
    info_relevante = buscar_con_rag(pregunta)
    if info_relevante and info_relevante != "No se encontró respuesta.":
        return jsonify({"respuesta": info_relevante})

    respuesta = generar_respuesta(pregunta, info_relevante)
    return jsonify({"respuesta": respuesta})

def buscar_respuesta_aproximada(pregunta, respuestas):
    mejor, score, _ = process.extractOne(pregunta, respuestas.keys())
    if score > 85:  # Puedes ajustar el umbral
        return respuestas[mejor]
    return None

if __name__ == "__main__":
    app.run(debug=True, port=5025)
    print("* Running on http://127.0.0.1:5025")
