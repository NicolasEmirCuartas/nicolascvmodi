import csv

# Datos proporcionados
datos = [
    ["Sección", "Información"],
    ["Nombre", "Nicolas Emir Cuartas"],
    ["Edad", "27 años"],
    ["Email", "emircuartas@gmail.com"],
    ["Cursos Realizados", "Universidad de Helsinki, Harvard, Microsoft: IA con Python"],
    ["Certificado", "Python - Universidad del Plata"],
    ["Carreras en Curso", "Licenciatura en Sistemas y Ciencia de Datos (no finalizadas)"],
    ["Experiencia Laboral", "Fundación: Gestión de datos de mercancías y ventas durante la pandemia"],
    ["Proyectos Tecnológicos", "Página web para un gimnasio (HTML, CSS, JavaScript), Aplicación para el club Universal de La Plata (Data Analyst para socios)"],
    ["Hobbies", "Basket: Técnico a cargo de un grupo de personas de 20 a 23 años"],
    ["Habilidades", "Adaptación al grupo, Manejo de la presión, Creatividad"],
    ["Resumen", "Soy un chico que intenta involucrarse en el mundo de las IA. Soy carismático, lo que me permite adaptarme perfectamente a los grupos. Estudiante de Licenciatura en Sistemas y Ciencia de Datos, me apasiona la programación. Comencé con un curso de Python en la Universidad del Plata y continué con cursos en Harvard, la Universidad de Helsinki y Microsoft sobre IA con Python. Durante la pandemia, gestioné datos de mercancías y ventas en la Cueva Solidaria. Desarrollé una página web para un gimnasio utilizando HTML, CSS y JavaScript. Fui entrenador de básquet para jóvenes de 20 a 23 años, liderando y gestionando datos relacionados. También trabajé en una financiera manejando Excel y en una empresa mejorando datos para su aplicación."],
    ["Pregunta", "¿Qué habilidades debo destacar para este puesto?"],
    ["Respuesta", "Para destacar en este puesto, deberías resaltar tus habilidades técnicas relacionadas con el rol, como el manejo de herramientas específicas o lenguajes de programación, así como tus habilidades blandas, como la comunicación efectiva, el trabajo en equipo y la resolución de problemas. Además, menciona cualquier experiencia previa relevante que demuestre tu capacidad para cumplir con las responsabilidades del puesto."]
]

# Función para guardar los datos en un archivo CSV
def guardar_en_csv(nombre_archivo, datos):
    try:
        with open(nombre_archivo, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(datos)
        print(f"Archivo CSV '{nombre_archivo}' creado exitosamente.")
    except Exception as e:
        print(f"Error al crear el archivo CSV: {e}")

# Llamada a la función
guardar_en_csv("infonico.csv", datos)

# Ubicación del archivo CSV
ubicacion_archivo = "D:/Escritorio/nicoalspyoectos/nicolasia/infonico.csv"
print(f"La ubicación del archivo CSV es: {ubicacion_archivo}")