from setuptools import setup

setup(
    name='Illumination_Prediction',
    packages=['Illumination_Prediction',
              'Illumination_Prediction.AutoEncoder',
              'Illumination_Prediction.IlluminationPredictor',
              'Illumination_Prediction.tools'],
    install_requires=['numpy', 'tqdm', 'visdom', 'pyyaml','openexr', 'torchvision', 'rotlib', 'imageio']
)