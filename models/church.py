import torch
from diffusers import DDPMPipeline, DDPMScheduler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class Model():
    """
    A class for hijacking DDPM-based image diffusion at timestep t > 0 using a pretrained model.

    Attributes:
        model_id (str): Identifier for the pretrained DDPM model from Hugging Face.
        device (str): The computation device, must be cuda for many Hugging Face models.
        scheduler (DDPMScheduler): The scheduler used to add and reverse noise.
        pipeline (DDPMPipeline): The full pipeline to process diffusion steps.
        image_tensor (torch.Tensor): The current loaded or processed image tensor.
        timestep (int): The timestep t used when adding synthetic noise.
        diffused (torch.Tensor): The image tensor after the diffusion process.
    """
    def __init__(self) -> None:
        """
        Initializes the DDPM model, scheduler, and pipeline.
        Loads the pretrained model to the appropriate device.
        """
        self.model_id = "google/ddpm-ema-church-256"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.scheduler = DDPMScheduler.from_pretrained(self.model_id)
        self.pipeline = DDPMPipeline.from_pretrained(self.model_id).to(self.device)
        self.image_tensor = None
        self.timestep = 200
        self.diffused = None
        
    def getImage(self):
        """
        Returns the currently loaded pre-diffusion image tensor.
        Will be None if no image set.

        Returns:
            torch.Tensor: The pre-diffusion image tensor.
        """
        return self.image_tensor

    def setImage(self, image):
        """
        Sets the internal image tensor.

        Args:
            image (torch.Tensor): The image tensor to set.
        """
        self.image_tensor = image
    
    def getDiffused(self):
        """
        Returns the image after diffusion has been performed. 
        Will be None if no diffusion has occured.

        Returns:
            torch.Tensor: The diffused image tensor.
        """
        return self.diffused

    def loadImage(self, image):
        """
        Loads and preprocesses a PIL image into a model-ready tensor.
        Prefered to setImage if no tensorized Image exists.

        Args:
            image (PIL.Image.Image): The input image.
        """
        transform = transforms.Compose([
            transforms.CenterCrop(min(image.size)),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        self.image_tensor = transform(image).unsqueeze(0).to(self.device)

    def loadImageFromFile(self, filename):
        """
        Loads an image from a file and preprocesses it.

        Args:
            filename (str): Path to the image file.
        """
        image = Image.open(filename).convert("RGB")
        self.loadImage(image)
    

    def normalizeImage(self):
        """
        Normalizes the image tensor to have zero mean and unit variance per channel,
        then rescales to [0, 1] for visualization.
        """
        batch, channels, height, width = self.image_tensor.shape
        flat = self.image_tensor.view(channels, -1)

        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)

        normalized = (flat - mean) / (std + 1e-8)

        scaled = normalized * 64 + 127.5
        scaled = scaled.clamp(0, 255) / 255 
        self.image_tensor = scaled.view_as(self.image_tensor)

    def setTimestep(self, timestep):
        """
        Sets the timestep used when adding noise.

        Args:
            timestep (int): Timestep value between 0 and 1000.
        """
        self.timestep = timestep

    def testSyntheticNoise(self):
        """
        Adds synthetic noise to the current image using the scheduler.

        Returns:
            int: -1 if image is not loaded, otherwise None.
        """
        if self.image_tensor == None:
            print("Ensure Image is Loaded")
            return -1
        noise = torch.randn_like(self.image_tensor)
        self.image_tensor = self.scheduler.add_noise(self.image_tensor, noise, torch.tensor([self.timestep], device=self.device))

    def displayTensor(self, tensor, title = "Image"):
        """
        Displays a tensor as an image using matplotlib.

        Args:
            tensor (torch.Tensor): Image tensor to display.
            title (str, optional): Title of the displayed image.
        """
        img = tensor.squeeze().permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.show()
    
    def displayNoisy(self):
        """
        Displays the noisy image before diffusion.
        """
        self.displayTensor(self.image_tensor, title=f"Noisy Image (Pre Diffusion), to be hijacked at t = {self.timestep}")

    def displayDiffused(self):
        """
        Displays the result after the diffusion process.
        """
        self.displayTensor(self.diffused, title=f"Diffusion Cleaned Image")
    
    def diffusion(self, verbose = False):
        """
        Runs the reverse diffusion process on the noisy image.

        Args:
            verbose (bool): Whether to print progress during diffusion.
        """
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
        """
        A generator version of the diffusion process for visualization or debugging.

        Args:
            verbose (bool): Whether to print progress.

        Yields:
            torch.Tensor: Intermediate image tensors at each step.
        """
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