# Breast Cancer Stage Detection Model

This is a machine learning model for detecting the stage of breast cancer. The model takes in several input parameters related to breast cancer, and outputs whether the cancer is in the benign (non-harmful) or malignant (dangerous) stage. 

## Dataset

The dataset used in this model is the Breast Cancer Wisconsin (Diagnostic) dataset, which is included in the `sklearn` Python package. The dataset contains 569 instances of breast cancer patients, with 30 different features related to each patient's tumor. The dataset has been preprocessed and cleaned, with missing values and outliers removed.

## Model Training and Evaluation

The dataset is split into training and testing sets using the `train_test_split` function from the `sklearn.model_selection` module. The split is done with a 90-10 ratio, with 90% of the data used for training and 10% used for testing. The `stratify` parameter is used to ensure that the training and testing sets have the same proportion of malignant and benign cases as the original dataset.

The logistic regression model is implemented using the `LogisticRegression` class from the `sklearn.linear_model` module. The model is trained on the training set using the `fit` method, and then evaluated on the testing set using the `accuracy_score` function from the `sklearn.metrics` module.

## Usage

To use the model, simply provide a list of input parameters related to a breast cancer patient. The model will output either "The Breast Cancer is Malignant" or "The Breast Cancer is Benign", indicating the stage of the cancer. 

The input parameters should be in the following order:
- radius_mean
- texture_mean
- perimeter_mean
- area_mean
- smoothness_mean
- compactness_mean
- concavity_mean
- concave points_mean
- symmetry_mean
- fractal_dimension_mean
- radius_se
- texture_se
- perimeter_se
- area_se
- smoothness_se
- compactness_se
- concavity_se
- concave points_se
- symmetry_se
- fractal_dimension_se
- radius_worst
- texture_worst
- perimeter_worst
- area_worst
- smoothness_worst
- compactness_worst
- concavity_worst
- concave points_worst
- symmetry_worst
- fractal_dimension_worst

The values should be normalized and scaled to the same range as the original dataset. 

## Collaborate

To explore the implementation details and code, you can visit the [Breast Cancer Stage Detection Colab Notebook](https://colab.research.google.com/drive/1P2EMic0LoSTwuRBmwqq40NLG9rB3451G?usp=sharing).

## Empowering Women's Health

Breast cancer is a serious health concern that affects millions of women worldwide. Early detection and timely treatment can significantly improve the chances of successful outcomes. It is crucial for women to prioritize their health and undergo regular screenings for the early detection of breast cancer.

We encourage women to:

- Perform regular breast self-examinations
- Schedule routine clinical breast examinations
- Follow recommended mammogram screening guidelines
- Stay informed about breast cancer risk factors and prevention strategies
- Support and encourage others in their journey to breast health

Together, we can make a difference in women's health and promote a future free from breast cancer.

## License

This project is licensed under the [MIT License](LICENSE).
