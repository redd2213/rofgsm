import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
from PIL import Image

def load_image(image_path):
    # Using pillow to open the needed image from disk
    image= Image.open(image_path).convert('RGB')

    # Keep track of the original image size for later use
    original_size = image.size

    # Definition of tensor conversion needed for PyTorch models
    transform = transforms.Compose ([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Transforming the original image just to tensor without resizing
    original_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Creating both tensors
    model_tensor = transform(image).unsqueeze(0)
    original_tensor = original_transform(image).unsqueeze(0)

    return original_tensor, model_tensor, original_size

def load_model():
    # Loadimg a pretrained ResNet50 model from torchvision
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Ensuring evaluation mode and not training mode
    model.eval()

    return model

def fgsm_attack(original_tensor, model_tensor, epsilon=0.05):
    # Load model amd get prediction from resized tensor
    model=load_model()
    with torch.no_grad():
        output = model(model_tensor)
    predicted_class = output.argmax(dim=1)

    # Tracking the gradients on the full size tensor
    original_tensor.requires_grad = True

    # Run the original tensor through the model so we don't need a resize for the gradient calculation
    output_original = model(torch.nn.functional.interpolate(
        original_tensor, size=(224, 224)
    ))

    # Calculating the loss
    loss = torch.nn.CrossEntropyLoss()(output_original, predicted_class)

    # Backpropagate
    model.zero_grad()
    loss.backward()

    # Getting the gradient sign from the original tensor
    gradient_sign=original_tensor.grad.sign()

    # Apply adversions at full res
    perturbed_tensor = original_tensor + epsilon * gradient_sign
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