# Instead of using mmagic, let's use diffusers which is more compatible with Python 3.12
import torch
from diffusers import DDPMPipeline, DDPMScheduler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class Model():
    def __init__(self) -> None:
        self.model_id = "google/ddpm-ema-church-256"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.scheduler = DDPMScheduler.from_pretrained(self.model_id)
        self.pipeline = DDPMPipeline.from_pretrained(self.model_id).to(self.device)
        self.image_tensor = None
        self.timestep = 200
        self.diffused = None
        
    def getImage(self):
        return self.image_tensor

    def setImage(self, image):
        self.image_tensor = image
    
    def getDiffused(self):
        return self.diffused

    def loadImage(self, image):
        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size)),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.image_tensor = transform(image).unsqueeze(0).to(self.device)

    def loadImageFromFile(self, filename):
        image = Image.open(filename).convert("RGB")
        self.loadImage(image)
    

    def normalizeImage(self):
        batch, channels, height, width = self.image_tensor.shape
        flat = self.image_tensor.view(channels, -1)

        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)

        normalized = (flat - mean) / (std + 1e-8)

        scaled = normalized * 64 + 127.5
        scaled = scaled.clamp(0, 255) / 255 
        self.image_tensor = scaled.view_as(self.image_tensor)

    def setTimestep(self, timestep):
        self.timestep = timestep

    def testSyntheticNoise(self):
        if self.image_tensor == None:
            print("Ensure Image is Loaded")
            return -1
        noise = torch.randn_like(self.image_tensor)
        self.image_tensor = self.scheduler.add_noise(self.image_tensor, noise, torch.tensor([self.timestep], device=self.device))

    def displayTensor(self, tensor, title = "Image"):
        img = tensor.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.show()
    
    def displayNoisy(self):
        self.displayTensor(self.image_tensor, title=f"Noisy Image (Pre Diffusion), to be hijacked at t = {self.timestep}")

    def displayDiffused(self):
        self.displayTensor(self.diffused, title=f"Diffusion Cleaned Image")
    
    def diffusion(self, verbose = False):
        x_t = self.image_tensor.clone()
        for t in range(self.timestep, 0, -1):
            if verbose and t % 20 == 0:
                print(f"Reverse Diffusion -- Timestep {t}")
            t_tensor = torch.tensor([t], device=self.device)
            with torch.no_grad():
                noise_pred = self.pipeline.unet(x_t, t_tensor).sample
            x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample
        if verbose:
            print("Diffusion Completed")
        self.diffused = x_t
    
    def diffusionGenerator(self, verbose = False):
        x_t = self.image_tensor.clone()
        for t in range(self.timestep, 0, -1):
            if verbose and t % 20 == 0:
                print(f"Reverse Diffusion -- Timestep {t}")
                yield x_t
            elif t % 20 == 0:
                yield x_t
            t_tensor = torch.tensor([t], device=self.device)
            with torch.no_grad():
                noise_pred = self.pipeline.unet(x_t, t_tensor).sample
            x_t = self.scheduler.step(noise_pred, t, x_t).prev_sample
        if verbose:
            print("Diffusion Completed")
        self.diffused = x_t