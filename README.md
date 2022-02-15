# Integrated-Vision-Inspection-System :camera:

![GitHub Contributors Image](https://contrib.rocks/image?repo=msf4-0/Integrated-Vision-Inspection-System-IVIS)

<a href="https://github.com/msf4-0/Integrated-Vision-Inspection-System/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/msf4-0/Integrated-Vision-Inspection-System.svg?color=blue">
</a>
<a href="https://github.com/msf4-0/Integrated-Vision-Inspection-System/releases">
    <img alt="Releases" src="https://img.shields.io/github/release/msf4-0/Integrated-Vision-Inspection-System?color=success" />
</a>
<a href="https://github.com/msf4-0/Integrated-Vision-Inspection-System/releases">
    <img alt="Downloads" src="https://img.shields.io/github/downloads/msf4-0/Integrated-Vision-Inspection-System/total.svg?color=success" />
</a>
<a href="https://github.com/msf4-0/Integrated-Vision-Inspection-System/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/msf4-0/Integrated-Vision-Inspection-System?color=blue" />
</a>
<a href="https://github.com/msf4-0/Integrated-Vision-Inspection-System/pulls">
    <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/msf4-0/Integrated-Vision-Inspection-System?color=blue" />
</a>
<br><br>

The ***Integrated-Vision-Inspection-System (IVIS)*** is a computer vision app that allows users to train/customize/deploy their own model to fit various application.

## Installation :computer:
1. This application is supported on both Windows and Linux OS. 
2. This application supports both navtive and docker installation. 
3. This application has version with GPU support and without GPU support.
4. Only installation on Ubuntu have been tested for Linux OS.
5. For Windows OS, native installation is recommended while Linux OS, docker installation is recommended. 

<br>

- [Windows Native Installation](https://github.com/msf4-0/Integrated-Vision-Inspection-System-IVIS/files/8069494/Native.installation.on.windows.system.docx)
- [Linux Native Installation](https://github.com/msf4-0/Integrated-Vision-Inspection-System-IVIS/files/8069286/IntegrateVisionSystemNativeInstall.Linux.docx)
- [Windows Docker Installation](https://github.com/msf4-0/Integrated-Vision-Inspection-System-IVIS/files/8069282/IntegrateVisionSystemDockerInstall.Windows.docx)
- [Linux Docker Installation](https://github.com/msf4-0/Integrated-Vision-Inspection-System-IVIS/files/8069255/IntegrateVisionSystemDockerInstall.Linux.docx)


## Basic User Guide :open_book:
***NOTE: For more detailed user guide, counsult the wiki.***
#### Application Log in
1. During first time log in, the application will prompt for an admin password change and a PostgreSQL password.
2. Once the application is fully setup, the normal username and password prompt will show up everytime the application is launched. 

### Project Creation
 1. After login, create a project at the ***Home*** page.
 2. Fill in the details and add a dataset using one of the options shown in the page.

![image](https://user-images.githubusercontent.com/99409981/154070190-5211f2cd-dea5-444a-a58f-0590475dd6d9.png)

 3. Once everything is filled up click ***Submit*** and the project will be created.

### Labelling
 1. During project creation, if the dataset selected was already labelled, this step can be skipped.
 2. if not, proceed to labelling and label the uploaded dataset.
 3. The labelling of this application uses the app ***LabelStudio*** interface.
 4. Due to some PC computation, loading time between each label can be slow, consider installing [Label Studio](https://labelstud.io/) natively and label there.

### Training 
 1. Once preparation of data is completed, navigate to the training section.
 2. Create a training session and follow the instruction in the application to setup the model, training parameters and image augmentation.
 3. When setting up the training parameters, beware of the PC resources and make sure it can handle.
 4. Start the training. Once the training is done, the model will be saved.
 5. Adjust the training parameters and continue the training if necessary.
 6. If desired, uploading a user model is possible as well.

### Deployment 
 1. Once a training is done, proceed to deployment page and select the desired model.
 2. Deploy the model and select the media for deployment. (image upload/camera/mqtt) 

### Settings
 1. This section will allow for deletion of object.
 2. This section allows the following deletion.
    - Training session/model
    - Dataset
    - Project

### User Management Page
 1. At ***Home*** page, the ***User Management*** option is located at the side tab.

![image](https://user-images.githubusercontent.com/99409981/154070530-64ee053b-1626-490d-b523-8d4d3e8c1523.png)

 2. This option allows addition/deletion/check info of a user.
 3. It is also possible to alter the user password in this option.

## Citation
```tex
@misc{Integrated Vision Inspection System,
  title={{Integrated Vision Inspection System}},
  url={https://github.com/msf4-0/Integrated-Vision-Inspection-System},
  author={
    Chu Zhen Hao,
    Anson,
    Yap Jun Kang,
    Lee Shi-Hau
    Nicholas Tan Chien Yuan},
  year={2021},
}
```
 
## License

This software is licensed under the [GNU GPLv3 LICENSE](/LICENSE) © [Selangor Human Resource Development Centre](http://www.shrdc.org.my/). 2021.  All Rights Reserved.
