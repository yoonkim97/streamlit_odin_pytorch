import streamlit as st
import cal
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from statistics import mode
from torch.autograd import Variable

st.title("Out-of-Distribution Image Detector")
criterion = nn.CrossEntropyLoss()
result = 0
# softmax_path = "/Users/yoonkim/streamlit_odin_pytorch/softmax_scores/"
# for file in os.listdir(softmax_path):
#     with open(softmax_path  + file, 'r') as file:
#         data = file.read()
#         st.text(data)

# ourIn = np.loadtxt("/Users/yoonkim/streamlit_odin_pytorch/softmax_scores/confidence_Our_In.txt", delimiter=',')
# ourInScores = ourIn[:, 2]
# sum_val = 0
# for i in range(len(ourInScores)):
#     sum_val += float(ourInScores[i])
# avg = float(sum_val / len(ourInScores))
# st.text(float(avg))
#
# st.text(mode(ourInScores))
st.header("Let's detect anomalies!")

st.subheader("Model")
path = "/home/yoon/jyk416/streamlit_odin_pytorch/models/"
models = []
for file in os.listdir(path):
    if file.lower().endswith("pth"):
        models.append(file)
model_name = st.selectbox("Select model:", models)

st.echo()
st.subheader("Temperature")
temperature = st.selectbox("Choose your temperature:", [1, 500, 1000])

st.subheader("Perturbation Magnitude")
perturbation_magn = st.selectbox("Choose your perturbation magnitude: ", [0.0000, 0.0008, 0.0032])

st.subheader("Your selection: ")
st.write('You selected model: `%s`' % model_name)
st.write('You selected temperature: `%d`' % temperature)
st.write('You selected perturbation magnitude `%f`' % perturbation_magn)

if model_name == 'model_no_cardiomegaly.pth' and temperature == 1 and perturbation_magn == 0.0000:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_1_magn_0.txt", delimiter=',')
if model_name == 'model_no_cardiomegaly.pth' and temperature == 1 and perturbation_magn == 0.0008:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_1_magn_8.txt", delimiter=',')
if model_name == 'model_no_cardiomegaly.pth' and temperature == 1 and perturbation_magn == 0.0032:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_1_magn_32.txt", delimiter=',')

if model_name == 'model_no_cardiomegaly.pth' and temperature == 500 and perturbation_magn == 0.0000:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_500_magn_0.txt", delimiter=',')
if model_name == 'model_no_cardiomegaly.pth' and temperature == 500 and perturbation_magn == 0.0008:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_500_magn_8.txt", delimiter=',')
if model_name == 'model_no_cardiomegaly.pth' and temperature == 500 and perturbation_magn == 0.0032:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_500_magn_32.txt", delimiter=',')

if model_name == 'model_no_cardiomegaly.pth' and temperature == 1000 and perturbation_magn == 0.0000:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_1000_magn_0.txt", delimiter=',')
if model_name == 'model_no_cardiomegaly.pth' and temperature == 1000 and perturbation_magn == 0.0008:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_1000_magn_8.txt", delimiter=',')
if model_name == 'model_no_cardiomegaly.pth' and temperature == 1000 and perturbation_magn == 0.0032:
    ourIn = np.loadtxt("/home/yoon/jyk416/streamlit_odin_pytorch/softmax_scores/temp_1000_magn_32.txt", delimiter=',')

ourInScores = ourIn[:, 2]
sum_val = 0
for i in range(len(ourInScores)):
    sum_val += float(ourInScores[i])
avg = float(sum_val / len(ourInScores))
threshold = max(mode(ourInScores), avg)

with st.spinner('Getting softmax scores of in-distribution images...'):
    time.sleep(5)
st.success('Done!')

st.subheader("Test Image Category")
category = st.radio("What category of image would you like to load?", options=['Healthy', 'Cat', 'Covid-19'])

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

if category == 'Healthy':
    filename = file_selector("/home/yoon/jyk416/streamlit_odin_pytorch/test/healthy-x-ray/healthy-x-ray-image/")
    folderpath = "/home/yoon/jyk416/streamlit_odin_pytorch/test/healthy-x-ray/"
elif category == 'Cat':
    filename = file_selector("/home/yoon/jyk416/streamlit_odin_pytorch/test/cat/cat_image/")
    folderpath = "/home/yoon/jyk416/streamlit_odin_pytorch/test/cat/"
else:
    filename = file_selector("/home/yoon/jyk416/streamlit_odin_pytorch/test/covid-x-ray/covid-x-ray-image/")
    folderpath = "/home/yoon/jyk416/streamlit_odin_pytorch/test/covid-x-ray/"

st.write('You selected `%s`' % filename)
image = Image.open(filename)
st.image(image, caption='Uploaded Image', use_column_width=True)
print(folderpath)


transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])

train_images = torchvision.datasets.ImageFolder(folderpath, transform=transform)
image_loader = torch.utils.data.DataLoader(train_images, batch_size=1, shuffle=False, num_workers=2)

model = cal.DenseNetBC_50_12()
model.load_state_dict(torch.load("/home/yoon/jyk416/streamlit_odin_pytorch/models/" + model_name, map_location=torch.device('cpu')))

def process_test_image(temper, magn):
    t0 = time.time()
    ###################################Out-of-Distributions#####################################
    for j, data in enumerate(image_loader):
        # if j<1000: continue
        images, _ = data

        inputs = Variable(images, requires_grad=True)
        outputs = model(inputs)

        # Calculating the confidence of the output, no perturbation added here
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))

        # Using temperature scaling
        outputs = outputs / temper

        # Calculating the perturbation we need to add, that is,
        # the sign of gradient of cross entropy loss w.r.t. input
        maxIndexTemp = np.argmax(nnOutputs)
        labels = Variable(torch.LongTensor([maxIndexTemp]))
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = (torch.ge(inputs.grad.data, 0))
        gradient = (gradient.float() - 0.5) * 2
        # Normalizing the gradient to the same space of image
        gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
        gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
        gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
        # Adding small perturbations to cat_image
        tempInputs = torch.add(inputs.data, -magn, gradient)
        outputs = model(Variable(tempInputs))
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs[0]
        nnOutputs = nnOutputs - np.max(nnOutputs)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
        result = float(np.max(nnOutputs))
        # if j % 100 == 99:
        print("{:4}/{:4} image processed, {:.1f} seconds used.".format(j + 1, 1, time.time() - t0))
        t0 = time.time()
    return result
    # if j== N-1: break



st.subheader("Processing Test Image...")
if category == 'Healthy':
    my_bar = st.progress(0)
    result = ourInScores[2]
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    if result >= threshold:
        st.success('THIS IS NOT AN OUTLIER.')
    else:
        st.warning('THIS IS AN OUTLIER.')
else:
    results = []
    my_second_bar = st.progress(0)
    current_progress = 1
    for i in range(3):
        result = process_test_image(temperature, perturbation_magn)
        my_second_bar.progress(current_progress + 33)
        current_progress += 33
        results.append(result)

    count = 0
    for val in results:
        if val <= threshold:
            count += 1
    if count >= 2:
        st.warning('THIS IS AN OUTLIER.')
    else:
        st.success('THIS IS NOT AN OUTLIER.')