from ultralytics import YOLO
model = YOLO(!Seu Caminho para o Modelo Aqui!)
results = model.predict(!Caminho para imagem Aqui!)
if isinstance(results, list):
    for result in results:
        result.show()  # Mostrar cada resultado individualmente
else:
    results.show()  # Mostrar o resultado se não for uma lista