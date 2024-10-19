import torch
from torchvision import transforms


def predict_with_probabilities(pil_image, model, device, index_to_label):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
        prob_dict = {index_to_label[idx]: float(prob) for idx, prob in enumerate(probs)}
        max_class = max(prob_dict, key=prob_dict.get)
        max_prob = prob_dict[max_class]

        return {
            'probabilities': {'trait': prob_dict},
            'predictions': {'trait': {
                'max_class': max_class,
                'max_prob': max_prob
            }},
            'final_classes': [max_class]
        }
