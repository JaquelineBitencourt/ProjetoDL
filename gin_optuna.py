import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch import nn
from torch_geometric.data import DataLoader, Data
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from torch_geometric.nn import BatchNorm, GINConv, global_mean_pool, global_add_pool
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import os
import logging
from copy import deepcopy
import joblib
from datetime import datetime
import gc
import psutil
import multiprocessing as mp
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get the number of CPUs allocated by SLURM (or use available CPUs if not in SLURM)
n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', mp.cpu_count()))

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# Early Stopping
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0

    def should_stop(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

# Function to convert SMILES to PyTorch Geometric graph - remains the same
def smiles_to_graph(smiles, labels):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    node_features = torch.tensor([
        [atom.GetAtomicNum(), int(atom.GetHybridization()), int(atom.GetIsAromatic()),
         atom.GetFormalCharge(), atom.GetTotalNumHs(), atom.GetDegree(), int(atom.IsInRing())]
        for atom in mol.GetAtoms()
    ], dtype=torch.float)

    edge_index = torch.tensor([], dtype=torch.long).view(2, 0)  # Caso sem arestas
    if mol.GetNumBonds() > 0:
        edge_index = torch.tensor([
            [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
            for bond in mol.GetBonds()
        ] + [
            [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
            for bond in mol.GetBonds()
        ], dtype=torch.long).t().contiguous()

    # Keep the original dimensionality to match what the model expects
    labels = torch.tensor(np.array(labels, dtype=np.float64), dtype=torch.float).unsqueeze(0)

    # Garantia de que global_features está sempre presente
    try:
        molecular_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        num_hba = Descriptors.NumHAcceptors(mol)
        num_hbd = Descriptors.NumHDonors(mol)
        global_features = torch.tensor([molecular_weight, logp, num_rotatable_bonds, num_hba, num_hbd], dtype=torch.float)
    except:
        global_features = torch.zeros(5, dtype=torch.float)  # Valor padrão caso falhe

    return Data(
        x=node_features,
        edge_index=edge_index,
        y=labels,
        global_features=global_features,
        smiles=smiles
    )


# GIN Layer implementation
class GINLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, eps=0, train_eps=False):
        super(GINLayer, self).__init__()
        
        # Multi-layer perceptron for the node transformation
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Initialize epsilon (either as a parameter or a fixed value)
        if train_eps:
            self.eps = nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
            
        self.gin_conv = GINConv(self.mlp, train_eps=train_eps, eps=eps)
        self.bn = BatchNorm(hidden_dim)
            
    def forward(self, x, edge_index):
        # Apply GIN convolution
        x = self.gin_conv(x, edge_index)
        x = self.bn(x)
        return x


# GIN Model Definition
class GINModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, 
                 mlp_hidden_dim, mlp_layers, global_dim, eps=0, train_eps=False):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        
        # GIN Layers
        self.gin_layers = nn.ModuleList()
        
        # First layer
        self.gin_layers.append(GINLayer(input_dim, hidden_dim, eps, train_eps))
        
        # Additional layers
        for _ in range(num_layers - 1):
            self.gin_layers.append(GINLayer(hidden_dim, hidden_dim, eps, train_eps))
            
        # MLP for classification after global pooling and concatenation with global features
        mlp_modules = [nn.Linear(hidden_dim + global_dim, mlp_hidden_dim), nn.LeakyReLU()]
        for _ in range(mlp_layers - 1):
            mlp_modules.extend([nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.LeakyReLU()])
        mlp_modules.append(nn.Linear(mlp_hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_features = getattr(data, 'global_features', None)
        
        # Apply GIN layers
        for gin_layer in self.gin_layers:
            x = gin_layer(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling (sum pooling is typically used with GIN)
        x = global_add_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Handle global features
        if global_features is None:
            global_features = torch.zeros((x.shape[0], 5), device=x.device)
        else:
            global_features = global_features.view(x.shape[0], -1)
        
        # Concatenate graph embeddings with global features
        x = torch.cat([x, global_features], dim=1)
        
        # Apply final MLP
        return self.mlp(x)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def objective(trial, train_dataset, val_dataset):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object for hyperparameter suggestion
        train_dataset: Dataset for training
        val_dataset: Dataset for validation
        
    Returns:
        float: AUC-ROC score on validation set
    """
    # Set unique seed for this trial to ensure reproducibility
    # but allow different trials to explore different initialization
    trial_seed = 42 + trial.number
    torch.manual_seed(trial_seed)
    np.random.seed(trial_seed)
    
    # Get worker info if running in parallel
    worker_id = 0
    if hasattr(trial, "_worker_id"):
        worker_id = trial._worker_id
    
    # Set different GPU for different workers if multiple GPUs are available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        gpu_id = worker_id % torch.cuda.device_count()
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Sample hyperparameters
    model_params = {
        "hidden_dim": trial.suggest_int("hidden_dim", 32, 256),
        "num_layers": trial.suggest_int("num_layers", 1, 4),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "mlp_hidden_dim": trial.suggest_int("mlp_hidden_dim", 64, 512),
        "mlp_layers": trial.suggest_int("mlp_layers", 1, 3),
        "eps": trial.suggest_float("eps", 0.0, 0.5),
        "train_eps": trial.suggest_categorical("train_eps", [True, False])
    }
    
    optimizer_params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    }
    
    # Create data loaders - using pre-split training and validation sets to avoid leakage
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Determine output dimension
    y_shape = train_dataset[0].y.shape
    output_dim = y_shape[1]  # Labels are in shape [1, num_labels]
    
    # Initialize model
    model = GINModel(
        input_dim=train_dataset[0].x.shape[1],
        hidden_dim=model_params["hidden_dim"],
        num_layers=model_params["num_layers"],
        output_dim=output_dim,
        dropout=model_params["dropout"],
        mlp_hidden_dim=model_params["mlp_hidden_dim"],
        mlp_layers=model_params["mlp_layers"],
        global_dim=train_dataset[0].global_features.shape[0],
        eps=model_params["eps"],
        train_eps=model_params["train_eps"]
    ).to(device)
    
    # Setup training
    optimizer = optim.AdamW(model.parameters(), **optimizer_params)
    criterion = nn.BCEWithLogitsLoss()
    
    # Track validation performance
    best_val_auc = 0.0
    patience = 15
    patience_counter = 0
    max_epochs = 50  # Reduced epochs for faster trials
    best_model_state = None
    
    # Training loop
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Handle batch_y dimensions
            batch_y = batch.y
            if batch_y.dim() > 2:
                batch_y = batch_y.squeeze(0)
            elif batch_y.dim() == 1:
                batch_y = batch_y.view(output.shape[0], output.shape[1])
            elif batch_y.shape[0] != output.shape[0]:
                batch_y = batch_y.view(output.shape[0], output.shape[1])
            
            # Calculate loss and backprop
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                
                # Handle batch_y dimensions
                batch_y = batch.y
                if batch_y.dim() > 2:
                    batch_y = batch_y.squeeze(0)
                elif batch_y.dim() == 1:
                    batch_y = batch_y.view(output.shape[0], output.shape[1])
                elif batch_y.shape[0] != output.shape[0]:
                    batch_y = batch_y.view(output.shape[0], output.shape[1])
                
                # Calculate validation loss
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                
                # Store predictions for metrics
                y_true.append(batch_y.cpu().numpy())
                y_pred.append(torch.sigmoid(output).cpu().numpy())
        
        # Calculate metrics
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        val_auc = roc_auc_score(y_true, y_pred, average='macro')
        
        # Average losses
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        # Report intermediate metric
        trial.report(val_auc, epoch)
        
        # Handle pruning based on Optuna threshold
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Track best validation performance using early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        # Log progress periodically
        if epoch % 5 == 0:
            logger.info(f"Trial {trial.number}, Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, AUC = {val_auc:.4f}")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Return best validation score
    return best_val_auc


def run_nested_cv(dataset, n_outer_splits=5, n_inner_splits=4, n_trials=20, timeout=None, n_jobs=1, output_dir=None):
    """
    Run nested cross-validation for hyperparameter tuning and model evaluation
    
    Args:
        dataset: List of torch_geometric.data.Data objects
        n_outer_splits: Number of outer CV splits for evaluation
        n_inner_splits: Number of inner CV splits for hyperparameter tuning
        n_trials: Number of trials per inner fold
        timeout: Maximum time per fold in seconds
        n_jobs: Number of parallel jobs for Optuna optimization
        output_dir: Directory to save results
        
    Returns:
        dict: Performance metrics and best hyperparameters for each outer fold
    """
    # At the beginning of run_nested_cv function
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    logger.info(f"SLURM allocated {n_cpus} CPUs for this job")

    # Then use n_cpus as your n_jobs parameter

    if output_dir is None:
        output_dir = os.path.join(os.getenv('HOME'), 'gin_optuna_results')
        os.makedirs(output_dir, exist_ok=True)
    # Extract labels for stratification
    labels = np.array([data.y.squeeze(0).numpy() for data in dataset])
    
    # Initialize outer k-fold for evaluation
    outer_cv = MultilabelStratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=42)
    
    results = {
        "test_auc": [],
        "best_params": []
    }
    
    for outer_fold, (train_val_idx, test_idx) in enumerate(outer_cv.split(dataset, labels)):
        logger.info(f"\n===== Outer Fold {outer_fold+1}/{n_outer_splits} =====")
        
        # Split into train_val and test sets
        train_val_data = [dataset[i] for i in train_val_idx]
        test_data = [dataset[i] for i in test_idx]
        
        # Extract labels for inner stratification
        inner_labels = np.array([data.y.squeeze(0).numpy() for data in train_val_data])
        
        # Set up Optuna study for this outer fold
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        sampler = optuna.samplers.TPESampler(seed=42 + outer_fold)
        study_name = f"gin_optimization_fold_{outer_fold}"
        
        # Separate validation set for parameter selection only for this fold's tuning
        train_idx, inner_val_idx = train_test_split(
            np.arange(len(train_val_data)), 
            test_size=0.2,
            random_state=42 + outer_fold,
            stratify=inner_labels
        )
        
        inner_train_data = [train_val_data[i] for i in train_idx]
        inner_val_data = [train_val_data[i] for i in inner_val_idx]
        
        # Set up storage for this study (SQLite database)
        storage_path = os.path.join(output_dir, f"optuna_fold_{outer_fold}.db")
        storage = f"sqlite:///{storage_path}"
        
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            pruner=pruner,
            sampler=sampler,
            storage=storage,
            load_if_exists=True
        )
        
        # Run hyperparameter optimization on inner train/val split
        fold_start_time = datetime.now()
        logger.info(f"Starting hyperparameter optimization for fold {outer_fold+1} with {n_jobs} parallel workers...")
        
        # Wrap objective function to pass the data sets
        objective_with_data = lambda trial: objective(trial, inner_train_data, inner_val_data)
        
        # Run optimization with parallel jobs
        if n_jobs > 1:
            study.optimize(
                objective_with_data,
                n_trials=n_trials,
                timeout=timeout,
                gc_after_trial=True,
                n_jobs=n_cpus,
                show_progress_bar=True
            )
        else:
            study.optimize(
                objective_with_data,
                n_trials=n_trials,
                timeout=timeout,
                gc_after_trial=True,
                show_progress_bar=True
            )
            
        fold_end_time = datetime.now()
        fold_duration = fold_end_time - fold_start_time
        logger.info(f"Fold {outer_fold+1} optimization completed in {fold_duration}")
        
        # Clean up memory to prevent CUDA OOM errors
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get best hyperparameters from this fold's study
        logger.info(f"Fold {outer_fold+1} best trial: {study.best_trial.number}")
        logger.info(f"Fold {outer_fold+1} best validation AUC: {study.best_value:.4f}")
        
        # Extract best hyperparameters
        best_model_params = {
        "hidden_dim": study.best_params["hidden_dim"],
        "num_layers": study.best_params["num_layers"],
        "dropout": study.best_params["dropout"],
        "mlp_hidden_dim": study.best_params["mlp_hidden_dim"],
        "mlp_layers": study.best_params["mlp_layers"],
        "eps": study.best_params["eps"],
        "train_eps": study.best_params["train_eps"]
        }
        
        best_optimizer_params = {
            "lr": study.best_params["lr"],
            "weight_decay": study.best_params["weight_decay"]
        }
        
        # Save visualization plots for this fold
        try:
            # Create optimization history plot
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_optimization_history(study)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'fold_{outer_fold+1}_optimization_GINhistory.png'))
            plt.close()
            
            # Create parameter importance plot
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'fold_{outer_fold+1}_param_GINimportances.png'))
            plt.close()
            
            # Create parallel coordinate plot
            plt.figure(figsize=(12, 8))
            optuna.visualization.matplotlib.plot_parallel_coordinate(study)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'fold_{outer_fold+1}_parallel_GINcoordinate.png'))
            plt.close()
            
            # Create slice plot for most important parameters
            param_importances = optuna.importance.get_param_importances(study)
            top_params = list(param_importances.keys())[:min(4, len(param_importances))]
            
            if top_params:
                plt.figure(figsize=(12, 10))
                optuna.visualization.matplotlib.plot_slice(study, params=top_params)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'fold_{outer_fold+1}_slice_GINplot.png'))
                plt.close()
        except Exception as e:
            logger.error(f"Error creating visualization for fold {outer_fold+1}: {e}")
        
        # Store best parameters for this fold
        best_params = {
            "model": best_model_params,
            "optimizer": best_optimizer_params
        }
        results["best_params"].append(best_params)
        
        # Now evaluate the best model from this fold on the held-out test set
        logger.info(f"Evaluating best model on test set for fold {outer_fold+1}...")
        
        # Create data loaders for final evaluation
        # Use all train_val data to train final model with best hyperparameters
        train_val_loader = DataLoader(train_val_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        # Initialize model with best hyperparameters
        model = GINModel(
            input_dim=train_val_data[0].x.shape[1],
            global_dim=train_val_data[0].global_features.shape[0],
            output_dim=train_val_data[0].y.shape[1],
            **best_model_params
        ).to(device)
        
        # Train final model
        optimizer = optim.AdamW(model.parameters(), **best_optimizer_params)
        criterion = nn.BCEWithLogitsLoss()
        
        # Early stopping for final model
        early_stopping = EarlyStopping(patience=20)
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(100):  # More epochs for final model
            # Training phase
            model.train()
            epoch_loss = 0
            
            for batch in train_val_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                output = model(batch)
                
                # Handle batch_y dimensions consistently
                batch_y = batch.y
                if batch_y.dim() > 2:
                    batch_y = batch_y.squeeze(0)
                elif batch_y.dim() == 1:
                    batch_y = batch_y.view(output.shape[0], output.shape[1])
                elif batch_y.shape[0] != output.shape[0]:
                    batch_y = batch_y.view(output.shape[0], output.shape[1])
                
                # Calculate loss and backprop
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            # Create a small validation set from training data for early stopping
            # This is a practical approach though still introduces a bit of data leakage
            # A more robust approach would be to use a separate validation set
            # completely held out from the hyperparameter selection process
            indices = torch.randperm(len(train_val_data))[:len(train_val_data)//5]
            val_subset = [train_val_data[i] for i in indices]
            val_subset_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
            
            # Validation phase for early stopping
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_subset_loader:
                    batch = batch.to(device)
                    output = model(batch)
                    
                    # Handle batch_y dimensions consistently
                    batch_y = batch.y
                    if batch_y.dim() > 2:
                        batch_y = batch_y.squeeze(0)
                    elif batch_y.dim() == 1:
                        batch_y = batch_y.view(output.shape[0], output.shape[1])
                    elif batch_y.shape[0] != output.shape[0]:
                        batch_y = batch_y.view(output.shape[0], output.shape[1])
                    
                    # Calculate validation loss
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_subset_loader)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
            
            # Check early stopping
            if early_stopping.should_stop(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Fold {outer_fold+1} final training - Epoch {epoch+1}: Train Loss = {epoch_loss/len(train_val_loader):.4f}, Val Loss = {val_loss:.4f}")
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation on test set
        model.eval()
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                
                # Handle batch_y dimensions consistently
                batch_y = batch.y
                if batch_y.dim() > 2:
                    batch_y = batch_y.squeeze(0)
                elif batch_y.dim() == 1:
                    batch_y = batch_y.view(output.shape[0], output.shape[1])
                elif batch_y.shape[0] != output.shape[0]:
                    batch_y = batch_y.view(output.shape[0], output.shape[1])
                
                y_true.append(batch_y.cpu().numpy())
                y_pred.append(torch.sigmoid(output).cpu().numpy())
        
        # Calculate final test metrics
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        test_auc = roc_auc_score(y_true, y_pred, average='macro')
        
        logger.info(f"Fold {outer_fold+1} Test AUC: {test_auc:.4f}")
        results["test_auc"].append(test_auc)
        
        # Save fold results visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.title(f"Optimization History - Fold {outer_fold+1}")
        
        plt.subplot(1, 2, 2)
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.title(f"Parameter Importance - Fold {outer_fold+1}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.getenv('HOME'), f'optuna_fold_{outer_fold+1}_GINresults.png'))
    
    # Overall results
    mean_test_auc = np.mean(results["test_auc"])
    std_test_auc = np.std(results["test_auc"])
    
    logger.info(f"\n===== Nested Cross-Validation Summary =====")
    logger.info(f"Test AUC: {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    
    # Calculate average best parameters
    avg_params = {
        "model": {},
        "optimizer": {}
    }
    
    # Handle numeric parameters
    for param in results["best_params"][0]["model"]:
        if param != "train_eps":  # Skip boolean parameters
            avg_params["model"][param] = np.mean([fold["model"][param] for fold in results["best_params"]])
    
    for param in results["best_params"][0]["optimizer"]:
        avg_params["optimizer"][param] = np.mean([fold["optimizer"][param] for fold in results["best_params"]])
    
    # Handle boolean parameters (majority vote)
    train_eps_values  = [fold["model"]["train_eps"] for fold in results["best_params"]]
    avg_params["model"]["train_eps"] = (sum(1 for v in train_eps_values if v) > len(train_eps_values) / 2)
    
    # Round integer parameters
    for param in ["hidden_dim", "num_layers", "mlp_hidden_dim", "mlp_layers"]:
        if param in avg_params["model"]:
            avg_params["model"][param] = int(round(avg_params["model"][param]))
    
    # Save overall results
    with open(os.path.join(output_dir, 'gin_nested_cv_results.txt'), 'w') as f:
        f.write("===== Nested Cross-Validation Results =====\n\n")
        f.write(f"Overall Test AUC: {mean_test_auc:.4f} ± {std_test_auc:.4f}\n\n")
        
        f.write("Results by Fold:\n")
        for i, auc in enumerate(results["test_auc"]):
            f.write(f"Fold {i+1}: AUC = {auc:.4f}\n")
        
        f.write("\nBest Parameters by Fold:\n")
        for i, params in enumerate(results["best_params"]):
            f.write(f"\nFold {i+1}:\n")
            f.write("  Model Parameters:\n")
            for param, value in params["model"].items():
                f.write(f"    {param}: {value}\n")
            f.write("  Optimizer Parameters:\n")
            for param, value in params["optimizer"].items():
                f.write(f"    {param}: {value}\n")
        
        f.write("\nAverage Best Parameters:\n")
        f.write("  Model Parameters:\n")
        for param, value in avg_params["model"].items():
            f.write(f"    {param}: {value}\n")
        f.write("  Optimizer Parameters:\n")
        for param, value in avg_params["optimizer"].items():
            f.write(f"    {param}: {value}\n")
    
    # Also save as JSON for easier programmatic access
    import json
    
    result_dict = {
        "overall_auc_mean": float(mean_test_auc),
        "overall_auc_std": float(std_test_auc),
        "fold_results": [float(auc) for auc in results["test_auc"]],
        "best_params_by_fold": results["best_params"],
        "average_best_params": avg_params
    }
    
    with open(os.path.join(output_dir, 'gin_nested_cv_results.json'), 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    # Print final recommended parameters
    print("\nRecommended hyperparameters for final model:")
    print("MODEL_HYPERPARAMS = {")
    for k, v in avg_params["model"].items():
        print(f'    "{k}": {v},')
    print("}")
    print("\nOPTIMIZER_HYPERPARAMS = {")
    for k, v in avg_params["optimizer"].items():
        print(f'    "{k}": {v},')
    print("}")
    
    return results, avg_params


def final_model_training(dataset, best_params, output_dir=None):
    """
    Train final model using the entire dataset with best hyperparameters
    
    Args:
        dataset: List of torch_geometric.data.Data objects
        best_params: Dictionary with best hyperparameters
        output_dir: Directory to save model and results
        
    Returns:
        trained model
    """
    if output_dir is None:
        output_dir = os.path.join(os.getenv('HOME'), 'gin_model_output')
        os.makedirs(output_dir, exist_ok=True)
    # Split dataset into train/test for final evaluation
    labels = np.array([data.y.squeeze(0).numpy() for data in dataset])
    train_idx, test_idx = train_test_split(
        np.arange(len(dataset)), 
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    train_data = [dataset[i] for i in train_idx]
    test_data = [dataset[i] for i in test_idx]
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    
    # Initialize model with best hyperparameters
    model = GINModel(
        input_dim=dataset[0].x.shape[1],
        global_dim=dataset[0].global_features.shape[0],
        output_dim=dataset[0].y.shape[1],
        **best_params["model"]
    ).to(device)
    
    # Train final model
    optimizer = optim.AdamW(model.parameters(), **best_params["optimizer"])
    criterion = nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=20)
    
    logger.info("Training final model with best hyperparameters...")
    
    # Training loop
    for epoch in range(200):  # More epochs for final model
        # Training phase
        model.train()
        epoch_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(batch)
            
            # Handle batch_y dimensions consistently
            batch_y = batch.y
            if batch_y.dim() > 2:
                batch_y = batch_y.squeeze(0)
            elif batch_y.dim() == 1:
                batch_y = batch_y.view(output.shape[0], output.shape[1])
            elif batch_y.shape[0] != output.shape[0]:
                batch_y = batch_y.view(output.shape[0], output.shape[1])
            
            # Calculate loss and backprop
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = model(batch)
                
                # Handle batch_y dimensions consistently
                batch_y = batch.y
                if batch_y.dim() > 2:
                    batch_y = batch_y.squeeze(0)
                elif batch_y.dim() == 1:
                    batch_y = batch_y.view(output.shape[0], output.shape[1])
                elif batch_y.shape[0] != output.shape[0]:
                    batch_y = batch_y.view(output.shape[0], output.shape[1])
                
                # Calculate validation loss
                loss = criterion(output, batch_y)
                val_loss += loss.item()
                
                # Store predictions for metrics
                y_true.append(batch_y.cpu().numpy())
                y_pred.append(torch.sigmoid(output).cpu().numpy())
        
        # Calculate metrics
        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        val_auc = roc_auc_score(y_true, y_pred, average='macro')
        
        # Average losses
        train_loss = epoch_loss / len(train_loader)
        val_loss = val_loss / len(test_loader)
        
        # Log progress
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, AUC = {val_auc:.4f}")
        
        # Early stopping
        if early_stopping.should_stop(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            output = model(batch)
            
            # Handle batch_y dimensions consistently
            batch_y = batch.y
            if batch_y.dim() > 2:
                batch_y = batch_y.squeeze(0)
            elif batch_y.dim() == 1:
                batch_y = batch_y.view(output.shape[0], output.shape[1])
            elif batch_y.shape[0] != output.shape[0]:
                batch_y = batch_y.view(output.shape[0], output.shape[1])
            
            y_true.append(batch_y.cpu().numpy())
            y_pred.append(torch.sigmoid(output).cpu().numpy())
    
    # Calculate final metrics
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Final metrics
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    test_auc = roc_auc_score(y_true, y_pred, average='macro')
    test_precision = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
    test_recall = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
    test_f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
    
    logger.info("\n===== Final Model Evaluation =====")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F1: {test_f1:.4f}")
    
    # Save model
    model_path = os.path.join(output_dir, 'final_gin_model.pt')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Final model saved to {model_path}")
    
    # Save model architecture and hyperparameters
    model_info = {
        "model_params": best_params["model"],
        "optimizer_params": best_params["optimizer"],
        "input_dim": dataset[0].x.shape[1],
        "global_dim": dataset[0].global_features.shape[0],
        "output_dim": dataset[0].y.shape[1],
        "metrics": {
            "test_auc": float(test_auc),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1)
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save as JSON
    import json
    with open(os.path.join(output_dir, 'model_gininfo.json'), 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return model


if __name__ == '__main__':
    # Load dataset - use the same code from the original script
    dataset_df = pd.read_csv('multi_label_dataset.csv')
    
    if 'smiles' not in dataset_df.columns:
        raise ValueError("The dataset must contain a 'smiles' column.")
    
    label_columns = [col for col in dataset_df.columns if col not in ['cid', 'smiles']]
    
    logger.info(f"Loading dataset with {len(dataset_df)} compounds and {len(label_columns)} labels")
    
    # Create graph dataset
    dataset = []
    for _, row in dataset_df.iterrows():
        smiles = row['smiles']
        labels = row[label_columns].tolist()
        graph = smiles_to_graph(smiles, labels)
        if graph is not None:
            dataset.append(graph)
    
    if not dataset:
        raise ValueError("No valid graphs were generated from the dataset.")
    
    logger.info(f"Created {len(dataset)} valid molecular graphs")
    
    # Ensure all data objects have global_features
    for i, data in enumerate(dataset):
        if not hasattr(data, 'global_features'):
            logger.warning(f"Warning: Data object {i} has no global_features! Replacing with zeros.")
            data.global_features = torch.zeros(5, dtype=torch.float)
    
    # Create output directory for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getenv('HOME'), f'gin_optuna_results_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Log system information
    logger.info(f"CPU cores: {psutil.cpu_count(logical=False)}, Threads: {psutil.cpu_count(logical=True)}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    logger.info(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    
    # Set the number of parallel jobs (adjust based on your CPU)
    n_jobs = min(20, psutil.cpu_count(logical=False))  # Using 20 cores max, adjust as needed
    logger.info(f"Using {n_jobs} cores for parallel optimization")
    
    # Run nested cross-validation with parallel processing
    results, avg_params = run_nested_cv(
        dataset,
        n_outer_splits=5,
        n_inner_splits=4,
        n_trials=50,  # 10x the number of hyperparameters for better exploration
        timeout=8*60*60,  # 8 hours per fold
        n_jobs=n_jobs,
        output_dir=output_dir
    )
    
    # Train final model with best hyperparameters
    final_model = final_model_training(dataset, avg_params, output_dir)
    
    # Calculate total runtime
    total_end_time = datetime.now()
    total_runtime = total_end_time - datetime.now()
    
    # Print summary
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Total runtime: {total_runtime}")
    print(f"Results saved to: {output_dir}")
    print(f"Best model saved to: {os.path.join(output_dir, 'final_gin_model.pt')}")
    print("\nRecommended hyperparameters for your model:")
    print("MODEL_HYPERPARAMS = {")
    for k, v in avg_params["model"].items():
        print(f'    "{k}": {v},')
    print("}")
    print("\nOPTIMIZER_HYPERPARAMS = {")
    for k, v in avg_params["optimizer"].items():
        print(f'    "{k}": {v},')
    print("}")
    print("="*50)