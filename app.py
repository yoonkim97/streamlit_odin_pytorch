import streamlit as st
import cal
import time
import calData
import calMetric
from densenet import DenseNet3
import main
import numpy as np
import os
import PIL
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from statistics import mode
from torch.autograd import Variable

st.title("Title")
criterion = nn.CrossEntropyLoss()

# softmax_path = "/Users/yoonkim/streamlit_odin_pytorch/softmax_scores/"
# for file in os.listdir(softmax_path):
#     with open(softmax_path  + file, 'r') as file:
#         data = file.read()
#         st.text(data)

# ourIn = np.loadtxt("/Users/yoonkim/streamlit_odin_pytorch/softmax_scores/confidence_Our_In.txt", delimiter=',')
# ourInScores = ourIn[:, 2]
#
# st.text(mode(ourInScores))

# path = "/Users/yoonkim/streamlit_odin_pytorch/models/"
# models =[]
# for file in os.listdir(path):
#     if file.lower().endswith("pth"):
#         models.append(file)
# model_name = st.radio("Select model:", models)

CUDA_DEVICE = 1
# MODEL
model = cal.DenseNetBC_50_12()
model.load_state_dict(torch.load("/home/yoon/jyk416/streamlit_odin_pytorch/models/model104.pth"))
optimizer1 = optim.SGD(model.parameters(), lr=0, momentum=0)
for i, (name, module) in enumerate(model._modules.items()):
    module = cal.recursion_change_bn(model)
model.cuda(CUDA_DEVICE)

transform_images = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])
train_no_cardiomegaly_dir = "/home/yoon/jyk416/streamlit_odin_pytorch/data/"
in_distr_dataset = torchvision.datasets.ImageFolder(train_no_cardiomegaly_dir, transform=transform_images)
in_distr_loader = torch.utils.data.DataLoader(in_distr_dataset, batch_size=1, shuffle=False, num_workers=2)

t0 = time.time()
N = 458
in_distr_base = []
in_distr_our = []
print("Processing in-distribution images")
########################################In-distribution###########################################
for j, data in enumerate(in_distr_loader):
    # if j<1000: continue
    images, _ = data

    inputs = Variable(images.cuda(CUDA_DEVICE), requires_grad=True)
    outputs = model(inputs)

    # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - np.max(nnOutputs)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
    in_distr_base.append(np.max(nnOutputs))

    # Using temperature scaling
    outputs = outputs / 1000

    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input
    maxIndexTemp = np.argmax(nnOutputs)
    labels = Variable(torch.LongTensor([maxIndexTemp]).cuda(CUDA_DEVICE))
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient = torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2
    # Normalizing the gradient to the same space of image
    gradient[0][0] = (gradient[0][0]) / (63.0 / 255.0)
    gradient[0][1] = (gradient[0][1]) / (62.1 / 255.0)
    gradient[0][2] = (gradient[0][2]) / (66.7 / 255.0)
    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data, -0.0002, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / 1000
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs[0]
    nnOutputs = nnOutputs - np.max(nnOutputs)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs))
    in_distr_our.append(np.max(nnOutputs))
    # if j % 100 == 99:
    print("{:4}/{:4} images processed, {:.1f} seconds used.".format(j + 1, N, time.time() - t0))
    t0 = time.time()

    if j == N - 1: break

t0 = time.time()
st.text(in_distr_our)

# artists = st.sidebar.multiselect("Select your model", models)
# temperature = st.sidebar.slider("Choose your temperature: ", min_value=1,
#                        max_value=1000, value=1, step=1)
# perturbation_magn = st.sidebar.selectbox("Choose your perturbation magnitude:",
#                                           np.arange(0.0000, 0.0040, 0.0002))
#
# uploaded_file = st.file_uploader("Choose an image...", type="jpg")
#
# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)