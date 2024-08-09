# SHAP-for-Image

### Overview
As black-box models grow increasingly complex, there is a growing need for effective explanation methods, with Shapley values emerging as a unified solution. This paper addresses the challenges of applying the SHAP algorithm to image data, given the high dimensionality and inherent correlations between pixels. We propose standardised guidelines for applying Shapley values in image classification tasks and conduct a series of experiments to evaluate the effects of superpixels, value functions, and link functions on explanation models. Our findings demonstrate that using sufficiently large superpixels is vital for preserving local accuracy. We also identify the most effective background replacement methods (value function) for images with single versus multiple objects. Additionally, our results suggest that probability space is more effective for highlighting the most significant superpixels, while log-odds space provides a more comprehensive view of all superpixels’ contributions, including both positive and negative impacts.

### Getting Started
To use this code, follow these steps:

1. **Clone this Repo**: 
    ```sh
   git clone https://github.com/kyphongmai/SHAP-for-Image.git
    ```
2. **Create a New Virtual Environment**:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```
3. **Install Requirements**:
   ```sh
   pip install -r requirements.txt
   ```

