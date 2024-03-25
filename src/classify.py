from src import train

class classifier:
    def __init__(self):
        if train.model_exists():
            self.model = train.create_model()
            self.load_labels()

    def load_labels(self):
        with open('labels.txt', 'r') as f:
            self.labels = f.read().split('\n')

    def classify(self, img):
        import torch
        from torchvision import transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(img)
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            self.model.eval()
            output = self.model(img.cuda())
            _, pred = torch.max(output, 1)

        return self.labels[pred.item()]
    
    def classify_topn(self, img, n):
        import torch
        from torchvision import transforms
        from PIL import Image

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(img)
        img = transform(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            self.model.eval()
            output = self.model(img.cuda())
            _, pred = torch.topk(output, n)

        return [self.labels[i] for i in pred[0].tolist()]
    
if __name__ == '__main__':
    import sys
    c = classifier()
    img = sys.argv[1]
    print(c.classify(img))
