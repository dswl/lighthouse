from models.church import Model

def main():
    model = Model()
    # testSynthetic(model, "./images/church2.jpg")
    model.setTimestep(175)
    testRealNoise(model, "./images/night.jpg")


def testSynthetic(model, filename):
    noisy_fname = filename
    model.loadImageFromFile(noisy_fname)
    model.normalizeImage()
    model.testSyntheticNoise(timestep=200)
    model.displayNoisy()
    model.diffusion(verbose=True)
    model.displayDiffused()

def testRealNoise(model, filename):
    noisy_fname = filename
    model.loadImageFromFile(noisy_fname)
    model.normalizeImage()
    model.displayNoisy()
    model.diffusion(verbose=True)
    model.displayDiffused()

main()