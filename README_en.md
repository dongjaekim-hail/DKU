# DKU
## Special Topics in Artificial Intelligence

### Model Introduction
- TabNet
  -  Deep learning model for structured data analysis that inherits the strengths of Deep Neural Network and DecisionTree-based model
  -  lthough it is difficult to say that it shows better performance than existing machine learning models, it is mainly used when the advantages of deep learning are needed rather than performance.
  -  Feature
    1) Flexible application to end-to-end learning with a structure that requires almost no preprocessing and uses gradient descent as an optimization algorithm
    2) Using sequential attention, it is possible to track the reason for feature selection -> secure interpretability (local & global)
  
  <img src=https://user-images.githubusercontent.com/59715960/234817143-c58d5125-1f07-49a5-af9d-1805c03a20ea.png />
  
  - Encoder  
<img src=https://user-images.githubusercontent.com/59715960/234817915-8102e9be-7526-4f6c-8a11-807eb9ec40c5.png width="600" height="300"/>
  
   - It is possible to identify the core features of each step by obtaining a feature selection mask for each step through various steps of the step (local)
   - It is possible to grasp the importance of features for all input data by summing the masks of all steps (global)
  
  - Result
    <img src=https://user-images.githubusercontent.com/59715960/235048302-64b58d87-aabb-4ac0-a349-17ff95f7c836.png width="400" height="300"/> 
    
### Requirements
- Python version 3.7.x, 3.8.x, 3.9.x, 3.10.x
- Tensorflow 2.x
(It was implemented with the pytorch-tabnet library, but implementation was performed using tensorflow)

### How to Run

### pseudocode
  - Sparse max
  
    <img width="300" alt="스크린샷 2023-05-12 오후 1 46 21" src="https://github.com/KR-ESWord/DKU/assets/59715960/dc63986d-0f35-4c96-969b-5811484d81f0">
    
    <pre>
    <code>
    def sparsemax(z):
      sum_all_z = sum(z)
      z_sorted = sorted(z, reverse=True)
      k = np.arange(len(z))
      k_array = 1 + k * z_sorted
      z_cumsum = np.cumsum(z_sorted) - z_sorted
      k_selected = k_array > z_cumsum
      k_max = np.where(k_selected)[0].max() + 1
      threshold = (z_cumsum[k_max-1] - 1) / k_max
      return np.maximum(z-threshold, 0)
    </code>
    </pre>

### To Do
  - Visualize Mask
