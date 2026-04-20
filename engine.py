import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

def load_image(image_path):
    # Using pillow to open the needed image from disk
    image= Image.open(image_path).convert('RGB')

    # Definition of tensor conversion needed for PyTorch models
    transform = transforms.Compose ([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Applying the defined transformation to the image and adding a batch dimension
    tensor = transform(image).unsqueeze(0)

    return tensor

def load_model():
    # Loadimg a pretrained ResNet50 model from torchvision
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Ensuring evaluation mode and not training mode
    model.eval()

    return model

def fgsm_attack(image_tensor, epsilon=0.01):
    # PyTorch tracks the gradients for this tensor
    image_tensor.requires_grad = True

    # Loading the pretrained model and recieving a prediction for the input image
    model= load_model()
    output = model(image_tensor)

    # The predicted class is the one with the highest score
    predicted_class = output.argmax(dim=1)

    # Calculating the loss
    loss = torch.nn.CrossEntropyLoss()(output, predicted_class)

    # Backpropagating the loss to get the gradients of the input image
    model.zero_grad()
    loss.backward()

    # Collecting the sign of the gradients
    gradient_sign=image_tensor.grad.sign()

    # Creating the adversarial image
    perturbed_tensor = image_tensor + epsilon * gradient_sign

    # Clamping values to valid image range
    perturbed_tensor=torch.clamp(perturbed_tensor, 0, 1)

    return perturbed_tensor

def save_image(perturbed_tensor, output_path):
    # Removing the batch dimension
    tensor = perturbed_tensor.squeeze(0)

    # Converting the tensor back to a PIL image
    transform = transforms.ToPILImage()
    image = transform(tensor)

    # Saving the adversarial image to disk
    image.save(output_path)