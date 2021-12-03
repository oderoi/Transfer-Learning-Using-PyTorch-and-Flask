import os
import io
from PIL import Image

import torch
import torchvision
from torchvision import transforms


from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

upload_folder="/home/isack/Desktop/PyCon Tanzania 2021/Flask_API/static"
model_path = "/home/isack/Desktop/PyCon Tanzania 2021/saved_model/resnet152.pth"
class_map = {
    0 : "Ant",
    1: "Bee"
}

def transform_image(image_byte):
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(225),
        transforms.CenterCrop(224),
        transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225])
    ])

    image=Image.open(io.BytesIO(image_byte))
    return transform(image).unsqueeze(0)

def prediction(model_path, image_byte, class_map):
    input = transform_image(image_byte=image_byte)
    
    model = torch.load(model_path)
    model.eval()

    output = model(input)
    _,pred = torch.max(output, 1)
    class_idx = pred.item()
    class_name = class_map[class_idx]

    return (class_idx, class_name)


@app.route("/", methods=["POST", "GET"])
def predict():

    if request.method== "POST":
        file = request.files["image"]

        if file:
            file_loc = os.path.join(
                upload_folder, file.filename
            )
            file.save(file_loc)

            with open(file_loc, 'rb') as f:
                img_byte = f.read()
                class_idx, class_name = prediction(model_path=model_path, image_byte=img_byte, class_map=class_map)


            return render_template("result.html", class_idx= class_idx, class_name =class_name, file_loc= file_loc, file_name=file.filename)

       
    return render_template ("index.html", class_idx= "None", class_name =None, file_loc= None)

    

if __name__=="__main__":
    app.run(port=5000, debug=True)


