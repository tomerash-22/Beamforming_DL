### **CNN_model_database**
---

### **`symmetric_matrix_from_params`**

* **Input:**

  * `params`: Tensor of shape `[batch, 7]` or `[7]` containing parameters `[a, b, c, d, e, f, g]`.
* **Output:**

  * `mat`: Complex symmetric 4×4 matrix (or batch of matrices) with 1s or `g` on the diagonal.

---

### **`MATLABDataset`**

* **Input:**

  * `mat_file_path`: Path to `.mat` dataset file containing `dataset_param_coupling` structure.
* **Output:**

  * PyTorch `Dataset` returning `(input_sample, target_sample)` each of shape `(2, 4, 4)` normalized by Frobenius norm.

---

### **`ResidualBlockWithoutBN`**

* **Input:**

  * `x`: Tensor input of shape `(batch, channels, H, W)`.
* **Output:**

  * Residual output tensor of same shape after Conv2D + ReLU + skip connection.

---

### **`ConvBlock`**

* **Input:**

  * `x`: Tensor `(batch, in_channels, H, W)`.
* **Output:**

  * Tensor after Conv2D + ReLU activation.

---

### **`ConvBlock_Wbn`**

* **Input:**

  * `x`: Tensor `(batch, in_channels, H, W)`.
* **Output:**

  * Tensor after Conv2D + BatchNorm + ReLU.

---

### **`ResidualBlock`**

* **Input:**

  * `x`: Tensor `(batch, in_channels, H, W)`.
* **Output:**

  * Residual tensor after Conv2D + BatchNorm + ReLU + skip connection.

---

### **`MyRealCNN`**

* **Input:**

  * `x`: Tensor `(batch, 2, 4, 4)` representing real/imaginary parts of input matrix.
* **Output:**

  * `result`: Tensor `(batch, 2, 4, 4)` — predicted complex matrix product result.
  * `Z_inv`: Tensor `(batch, 2, 4, 4)` — predicted inverse (complex) matrix.

---

### **`Reduced_param_CNN`**

* **Input:**

  * `x`: Tensor `(batch, 2, 4, 4)` representing real/imaginary input matrix.
* **Output:**

  * `result`: Tensor `(batch, 2, 4, 4)` — predicted output matrix.
  * `Z_inv`: Tensor `(batch, 2, 4, 4)` — reconstructed symmetric complex matrix from 7 parameters.

---

Would you like me to also include **a one-line summary** for each (e.g., purpose) alongside the input/output?

### `couplingcnn.py`

#### **`objective`**

* **Input**:

  * `trial`: Optuna trial object
* **Output**:

  * `avg_val_metric`: Validation loss metric for Optuna optimization

#### **`test_model_on_mat`**

* **Input**:

  * `model_path` (str): Path to trained model (`.pth`)
  * `mat_input_path` (str): Path to `.mat` input data
  * `mat_output_path` (str): Path for saving output `.mat` data
  * `input_key` (str): Key for input matrix (default: `'R_coupled'`)
  * `output_key` (str): Key for saving output matrix (default: `'predicted_R_sig'`)
* **Output**:

  * Saves predictions in `.mat` format

#### **`condition_number`**

* **Input**:

  * `matrix`: Complex matrix (batch, M, N)
* **Output**:

  * `kappa`: Condition number of the matrix

#### **`custom_loss_function`**

* **Input**:

  * `output`: Model output (complex)
  * `target`: Ground truth (complex)
  * `reg_factor_H`: Hermitian regularization factor
  * `reg_kappa`: Condition number regularization factor
  * `target_kappa`: Target condition number
* **Output**:

  * Loss value (scaled-invariant MSE)

#### **`train_model`**

* **Input**:

  * `model`: Model to train
  * `train_loader`: Training data loader
  * `val_loader`: Validation data loader
  * `optimizer`: Optimizer
  * `threshold`: Early stopping threshold
  * `max_epochs`: Max number of epochs
  * `batch_sz`: Batch size
  * `gamma`: Scheduler parameter
  * `step_size`: Scheduler step size
  * `reg_diag`: Regularization factor for Hermitian loss
  * `kappa_reg`: Condition number regularization factor
* **Output**:

  * Training losses, validation losses, gradient norms during training


