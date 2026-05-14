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
    # Loading a pretrained ResNet50 model from torchvision
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Ensuring evaluation mode and not training mode
    model.eval()

    return model

def fgsm_attack(original_tensor, model_tensor, epsilon=0.05):
    # Load model and get prediction from resized tensor
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

def pgd_attack(original_tensor, model_tensor, epsilon=0.1, alpha=0.005, iterations=50):
    # Load model and get prediction from resized tensor
    model = load_model()
    with torch.no_grad():
        output = model(model_tensor)
    predicted_class = output.argmax(dim=1)

    # Random start within epsilon bounds (stronger than the original image)
    noise = torch.empty_like(original_tensor).uniform_(-epsilon, epsilon)
    perturbed_tensor = torch.clamp(original_tensor + noise, 0, 1).clone()

    for i in range(iterations):
        # Tracking the gradients on the full size tensor
        perturbed_tensor.requires_grad = True

        # Run through the model at req size (224x224)
        output_p = model(torch.nn.functional.interpolate(
            perturbed_tensor, size=(224, 224)
        ))

        # Loss calculation
        loss = torch.nn.CrossEntropyLoss()(output_p, predicted_class)

        # Backpropagate
        model.zero_grad()
        loss.backward()

        # Small step towards gradient direction
        with torch.no_grad():
            perturbed_tensor = perturbed_tensor + alpha * perturbed_tensor.grad.sign()

            # Clamping so it does not drift far from the original image
            perturbation = torch.clamp(perturbed_tensor - original_tensor, -epsilon, epsilon)
            perturbed_tensor = original_tensor + perturbation

            # Clamping to valid pixel range
            perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)

    return perturbed_tensor

def save_image(perturbed_tensor, output_path):
    # Removing the batch dimension
    tensor = perturbed_tensor.squeeze(0)

    # Converting the tensor back to a PIL image
    transform = transforms.ToPILImage()
    image = transform(tensor)

    # Saving the adversarial image to disk
    image.save(output_path)

def get_prediction(model_tensor):
    import json
    import urllib.request
    import os

    # Download labels if needed
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    if not os.path.exists("labels.json"):
        urllib.request.urlretrieve(url, "labels.json")

    with open("labels.json") as f:
        labels = json.load(f)

    model = load_model()
    with torch.no_grad():
        output = model(model_tensor)
    
    predicted_index = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1)[0][predicted_index].item()
    
    return labels[predicted_index], round(confidence * 100, 1)

def load_clip_model():
    import clip
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()
    return model, preprocess

def get_clip_prediction(image_path):
    import clip
    from PIL import Image

    model, preprocess = load_clip_model()

    image = preprocess(Image.open(image_path)).unsqueeze(0)

    # Art-relevant text labels to compare against
    labels = [
        "a painting", "a digital artwork", "an illustration",
        "a photograph", "a sketch", "a watercolor painting",
        "a comic book", "concept art", "anime art",
        "an oil painting", "a charcoal sketch", "a pencil drawing",
        "a pastel painting", "an acrylic painting", "a gouache painting",
        "street art", "a print", "a lithograph"
    ]

    import torch
    text = clip.tokenize(labels)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = (image_features @ text_features.T).softmax(dim=-1)

    best_idx = similarity.argmax().item()
    confidence = round(similarity[0][best_idx].item() * 100, 1)

    return labels[best_idx], confidence

def clip_attack(image_path, epsilon=0.4, alpha=0.002, iterations=500, decoy="anime art"):
    import clip
    from PIL import Image
    import torchvision.transforms as transforms

    # Load CLIP
    model, preprocess = load_clip_model()

    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Convert to tensor
    to_tensor = transforms.ToTensor()
    original_tensor = to_tensor(image).unsqueeze(0)

    # Get original CLIP embedding
    clip_input = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        original_embedding = model.encode_image(clip_input)
        original_embedding = original_embedding / original_embedding.norm(dim=-1, keepdim=True)

        # Multiple decoy targets averaged into one direction
        decoys = ["anime art", "an oil painting", "a watercolor painting", "concept art"]
        decoy_text = clip.tokenize(decoys)
        decoy_embeddings = model.encode_text(decoy_text)
        decoy_embeddings = decoy_embeddings / decoy_embeddings.norm(dim=-1, keepdim=True)
        decoy_embedding = decoy_embeddings.mean(dim=0, keepdim=True)
        decoy_embedding = decoy_embedding / decoy_embedding.norm(dim=-1, keepdim=True)

    # Random start
    noise = torch.empty_like(original_tensor).uniform_(-epsilon, epsilon)
    perturbed_tensor = torch.clamp(original_tensor + noise, 0, 1).clone()

    # CLIP normalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

    for i in range(iterations):
        perturbed_tensor = perturbed_tensor.detach().requires_grad_(True)

        # Resize for CLIP keeping gradient flow
        clip_size = torch.nn.functional.interpolate(
            perturbed_tensor, size=(224, 224), mode='bilinear', align_corners=False
        )

        # Normalize for CLIP
        clip_input_perturbed = (clip_size - mean) / std

        # Get perturbed embedding
        perturbed_embedding = model.encode_image(clip_input_perturbed)
        perturbed_embedding = perturbed_embedding / perturbed_embedding.norm(dim=-1, keepdim=True)

        # Push away from original, toward averaged decoy
        loss = (original_embedding * perturbed_embedding).sum() - (decoy_embedding * perturbed_embedding).sum()
        loss.backward()

        with torch.no_grad():
            perturbed_tensor = perturbed_tensor - alpha * perturbed_tensor.grad.sign()
            perturbation = torch.clamp(perturbed_tensor - original_tensor, -epsilon, epsilon)
            perturbed_tensor = original_tensor + perturbation
            perturbed_tensor = torch.clamp(perturbed_tensor, 0, 1)

    return perturbed_tensor, original_size

def get_clip_distribution(image_path):
    import clip
    from PIL import Image

    model, preprocess = load_clip_model()

    labels = [
        "a digital artwork", "an illustration", "concept art",
        "anime art", "a painting", "an oil painting",
        "a watercolor painting", "a pastel painting",
        "a charcoal sketch", "a pencil drawing", "a photograph"
    ]

    image = preprocess(Image.open(image_path)).unsqueeze(0)
    text = clip.tokenize(labels)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        similarity = (image_features @ text_features.T).softmax(dim=-1)[0]

    return [(label, round(similarity[i].item() * 100, 1)) for i, label in enumerate(labels)]