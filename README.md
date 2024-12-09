# Tennis Analytics Project

This project is designed to analyze tennis matches using state-of-the-art data processing, computer vision, and machine learning techniques. It provides insights into player performance, ball tracking, court detection, and more, all through an intuitive graphical interface.

https://github.com/user-attachments/assets/141da489-5392-4664-bf85-371de0543d7b

Data necessary to execute everything is provided on Kaggle: https://www.kaggle.com/datasets/benediktvoss/tennisanalytics

## Project Structure

In generel the project can be devided into two parts:

1. **Data Analytics:** In the first Folders 01-06 the entire process of data gathering, data preperation, modeling, and evaluation is shown. This is done to ensure full transperency and reproducability.
2. **GUI:** In folder *07_GUI* a User Intefrace is implemented where the dataset can be visually explored and the 

- **`00_Dataset`**: Contains raw and processed datasets.
- **`01_DataGathering`**: Scripts and tools for collecting match data.
- **`02_EDA`**: Exploratory Data Analysis (EDA) for initial insights.
- **`03_BallTracking`**: Ball tracking algorithms and models.
- **`04_CourtDetection`**: Court detection using computer vision.
- **`05_PlayerTracking`**: Player tracking models and visualizations.
- **`06_Analytics`**: Advanced analytics and insights generation.
- **`07_GUI`**: GUI implementation for user interaction.
- **`LICENSE`**: License information.
- **`requirements.txt`**: Python dependencies for the project. Python 3.12 was used for developemnt.

In general the files and folders contain a prefix numeration and should be inspected and executed in the correct order.
## Requirements

The code is developed and run on a windows machine, possible for other os there are some changes in file path handling.

Install the dependencies listed in `requirements.txt`:

Additionally, ensure you have the following requirements installed:

- **PyTorch**: For deep learning models. Install it from [PyTorch](https://pytorch.org/get-started/locally/).
- **CUDA**: For GPU acceleration. Ensure you have the appropriate CUDA version installed. Check compatibility with your PyTorch version [here](https://pytorch.org/get-started/locally/).

**It is highly reccomended to run this on a machine with CUDA installed, otherwise trainig and Inference of the neural networks will be significantly slower!**


## How to Run

### Clone the Repository
```bash
git clone <[repository_url](https://github.com/BenediktVoss/TennisAnalytics)>
cd Tennis-Analytics-Project
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch the GUI
```bash
python 07_GUI/app.py
```

## Examples

- **Example Inference Video:** Check out `Example_Inference.mp4` for a demonstration of the project in action.
![GUI](https://github.com/user-attachments/assets/b3f5fc8c-69ac-4127-a294-90eff5fe4ca8)


## License
This project is licensed under the terms of the `LICENSE` file. Ensure to read and comply with its terms for usage and distribution.

## Usage of AI
For the creation of this project multiple AI tools for enhanced software developement were used, like GitHub Copilot and ChatGPT. 

