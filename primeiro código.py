#Importando a YOLO da Ultralytics
from ultralytics import YOLO
# Definindo o modelo da YOLO a ser utilizado
model = YOLO("yolov8n.pt")
model.train(data=!Seu Caminho Aqui!, epochs=2, imgsz=640)
results = model.val(data =!Seu Caminho Aqui!, batch = 16, save = True )
results = model.predict(source = !Seu Caminho Aqui!, save = True)
model.save(!Seu Caminho Aqui!)