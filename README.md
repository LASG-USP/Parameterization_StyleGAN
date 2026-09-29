# Parameterization method of reservoir properties for ensemble-based data assimilation 
#                 using intermediate latent space of StyleGAN 

This research is the improvement the generation of image facies with geologic realism in the same time with better data assimilation, trough the StyleGAN model using intermediate latent space (w-space). The results demonstrated that all three models are highly efficient, with the StyleGAN2 model standing out for generating samples with geological realism and achieving excellent data matching in the cases studied.

Parameterization_StyleGAN is a python3 project providing tools for the parameterization of reservoir properties for data assimilation using ESMDA.


Version 1.0 - September 2026

<img src="https://github.com/LASG-USP/Parameterization_StyleGAN/blob/main/Fig.1.png" width="1200">

<img src="https://github.com/LASG-USP/Parameterization_StyleGAN/blob/main/Fig.2.png" width="1200">

<img src="https://github.com/LASG-USP/Parameterization_StyleGAN/blob/main/Fig.3.png" width="1200">

<img src="https://github.com/LASG-USP/Parameterization_StyleGAN/blob/main/Fig.4.png" width="1200">


## Installation from github

If you want to access the source code and potentially contribute. You should follow the following steps.

### 1. Download

Download SG from the [Github repository](https://github.com/LASG-USP/Parameterization_StyleGAN): green button "clone or download". Then, unzip it on your computer. 


### 2. Go in the directory

Once this has been done, open a Python prompt (like the Anaconda prompt), and go to your directory location (ex: 
`cd C:\Users\YourName\Documents\SG`).

### 3. Launch the local installation

After downloading and opening the folder on your computer, you will have three folders: GAN, VAE, VAE-GAN.

1 - For the VAE-GAN model, within the SG folder, simply run the following for each of the created cases:

a - Discrete Case: run VAE_GAN_disc.py

b - Continuous Case: run VAE_GAN_cont.py

*Note: For this, you will need to install the libraries listed in the code's imports.

After training, to perform data assimilation using ESMDA: run assimilation_VAE_GAN_disc.py (discrete case) or assimilation_VAE_GAN_cont.py (continuous case).

*Note: For this, you will need to install the IMEX simulator of CMG (Computer Modelling Group).


2 - For the Latent Diffusion model, within the SG folder, simply repeat the previous steps, only changing the model name:

Training: LD_disc.py or LD_cont.py

Assimilation: assimilation_LD_disc.py or assimilation_LD_cont.py


3 - And for the StyleGAN2 model, within the SG folder:

Training: SG_disc_z.py (z-space), SG_disc_w.py (w-space) or SG_cont_z.py (z-space), SG_cont_w.py (w-space)

Assimilation: assimilation_SG_disc_z.py (z-space) and assimilation_SG_disc_w.py (w-space) or 
              assimilation_SG_cont_z.py (z-space) and assimilation_SG_cont_w.py (w-space)


### 4. Download of datasets

https://drive.google.com/drive/folders/1zYR8KPVr0mY-MfObsjGaod0iMbWD5swn?usp=sharing


### 5. Download of simulation files
https://drive.google.com/drive/folders/18b5SooLqCjMeTuYGJQXmROEnnuttsMQO?usp=sharing


### OBS. 
Don't forget to place the datasets and simfiles folders inside each folder before running!


## Documentation

These models were implemented to perform a comparison and verify which one results in better image generation with geological realism and an better history matching.

Several metrics were used for this purpose. 

To measure the quality of the generated images: Fréchet Inception Distance (FID) and Fréchet Reservoir Distance (FRD). 

To verify the quality and geological realism of the generated samples: variogram (MSE), connectivity (MSE), histogram KL, PCA correlation, and MDS MMD (Maximum Mean Discrepancy).


## Reference

The SG package implements three deep learning models (VAE-GAN, LDM and StyleGAN2) for parameterization and data assimilation that were
investigated and discussed in:

SAMPAIO, M. A.; RANAZZI, P. H.; BLUNT, M. J., Parameterization method of reservoir properties for ensemble-based data assimilation using intermediate latent space of StyleGAN, submitted to Geoenergy Science and Engineering.
