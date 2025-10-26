
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


