# %%
#!pip install --quiet torch torch-geometric torchvision optuna numpy pandas scikit-learn matplotlib tqdm rdkit iterative-stratification seaborn
# %%
#from google.colab import drive
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from torch import nn
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import BatchNorm, SAGEConv, global_mean_pool, global_max_pool
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Model hyperparameters #Estou usando o que saiu no optuna#
MODEL_HYPERPARAMS = {
    "hidden_dim": 200,
    "num_layers": 4,
    "dropout": 0.19858352310707672,
    "mlp_hidden_dim": 247,
    "mlp_layers": 2
}

# Optimizer hyperparameters
OPTIMIZER_HYPERPARAMS = {
    "lr": 0.0012338612556267276,
    "weight_decay": 0.00019459988616868634
}


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

    # Ensure labels is always 2D (batch_size, num_labels)
    if isinstance(labels, list):
        labels = torch.tensor(np.array(labels, dtype=np.float64), dtype=torch.float)
    elif isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.float)
    
    if labels.dim() == 1:
        # For single-label case, make it 2D
        labels = labels.unsqueeze(0)
    
    # Consistent global_features extraction
    try:
        molecular_weight = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        num_hba = Descriptors.NumHAcceptors(mol)
        num_hbd = Descriptors.NumHDonors(mol)
        global_features = torch.tensor([molecular_weight, logp, num_rotatable_bonds, num_hba, num_hbd], dtype=torch.float)
    except:
        global_features = torch.zeros(5, dtype=torch.float)  # Default value if computation fails
    
    # Return the graph with consistent dimensions
    return Data(
        x=node_features,
        edge_index=edge_index,
        y=labels,
        global_features=global_features,
        smiles=smiles
    )


# Definição do modelo GraphSAGE
class GraphSAGEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, mlp_hidden_dim, mlp_layers, global_dim, concat=True, negative_slope=0.042859419676898734):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.concat = concat
        self.negative_slope = negative_slope

        # Primeira camada de GraphSAGE
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.bns.append(BatchNorm(hidden_dim))

        # Camadas intermediárias
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.bns.append(BatchNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Store dimensions for debugging
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.global_dim = global_dim
        self.mlp_input_dim = hidden_dim + global_dim
        
        # MLP aprimorada: Recebe hidden_dim + global_dim
        mlp_modules = [nn.Linear(self.mlp_input_dim, mlp_hidden_dim), nn.LeakyReLU(negative_slope=self.negative_slope)]
        for _ in range(mlp_layers - 1):
            mlp_modules.extend([nn.Linear(mlp_hidden_dim, mlp_hidden_dim), nn.LeakyReLU(negative_slope=self.negative_slope)])
        mlp_modules.append(nn.Linear(mlp_hidden_dim, output_dim))

        self.mlp = nn.Sequential(*mlp_modules)

        # Inicialização de pesos
        self.apply(self._init_weights)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        global_features = getattr(data, 'global_features', None)  # Verifica se global_features existe

        # Log shapes for debugging
        batch_size = batch.max().item() + 1 if batch is not None else 1
        
        # Passagem pelas camadas GraphSAGE
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.leaky_relu(x, negative_slope=self.negative_slope)
            x = self.dropout(x)

        # Pooling global
        x = global_mean_pool(x, batch)  # Shape: [batch_size, hidden_dim]
        
        # Check hidden representation shape
        if x.size(1) != self.hidden_dim:
            raise ValueError(f"Node embedding dimension mismatch: expected {self.hidden_dim}, got {x.size(1)}")

        # Verificação segura para `global_features`
        if global_features is None:
            global_features = torch.zeros((x.size(0), self.global_dim), device=x.device)  # Usa um tensor zerado seguro
        elif global_features.dim() == 1:
            # Handle the case where global_features is a 1D tensor for a single molecule
            global_features = global_features.unsqueeze(0).expand(x.size(0), -1)
        elif len(global_features.shape) > 1 and global_features.size(0) != x.size(0):
            # We need to properly assign global features to each graph in the batch
            if batch is not None:
                # Create a tensor of the same size as batch to store global features
                batch_global_features = []
                for i in range(batch_size):
                    # Find indices of nodes belonging to graph i
                    idx = (batch == i).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        # Add the global features for this graph to the list
                        graph_idx = min(i, global_features.size(0) - 1)  # Prevent index out of bounds
                        batch_global_features.append(global_features[graph_idx].unsqueeze(0))
                
                # Stack all global features for the batch
                if batch_global_features:
                    global_features = torch.cat(batch_global_features, dim=0)
                else:
                    global_features = torch.zeros((x.size(0), self.global_dim), device=x.device)
            else:
                # Fallback: just repeat the first global feature for all graphs
                global_features = global_features[0:1].repeat(x.size(0), 1)
        
        # Ensure global_features has the right dimension
        if global_features.size(1) != self.global_dim:
            # Resize if dimensions don't match
            global_features = torch.zeros((x.size(0), self.global_dim), device=x.device)

        # Concatenar features do grafo + features globais
        x = torch.cat([x, global_features], dim=1)
        
        # Final check before MLP
        if x.size(1) != self.mlp_input_dim:
            raise ValueError(f"MLP input dimension mismatch: expected {self.mlp_input_dim}, got {x.size(1)}")

        return self.mlp(x)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='leaky_relu', a=self.negative_slope)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def cross_validate(dataset, model_params, optimizer_params, n_splits=5):
    skf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Extract labels for stratification
    labels = np.array([data.y.numpy() if data.y.dim() == 1 else data.y.squeeze(0).numpy() for data in dataset])
    
    # Track metrics across folds
    fold_metrics = {
        'train_loss': [], 'val_loss': [], 'auc': [], 
        'precision': [], 'recall': [], 'f1': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        print(f"Fold {fold+1}/{n_splits}")
        
        # Create data loaders
        train_data = [dataset[i] for i in train_idx]
        test_data = [dataset[i] for i in test_idx]
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        
        # Determine output dimension from training data
        # Ensure consistent shape handling
        output_dim = train_data[0].y.shape[0] if train_data[0].y.dim() == 1 else train_data[0].y.shape[1]
        
        # Initialize model
        model = GraphSAGEModel(
            input_dim=train_data[0].x.shape[1],
            global_dim=train_data[0].global_features.shape[0],
            output_dim=output_dim,
            **model_params
        ).to(device)
        
        # Setup training
        optimizer = optim.AdamW(model.parameters(), **optimizer_params)
        criterion = nn.BCEWithLogitsLoss()
        early_stopping = EarlyStopping(patience=200)
        
        # Metrics tracking
        train_losses = []
        val_losses = []
        auc_scores = []
        
        for epoch in range(500):
            # Training phase
            model.train()
            epoch_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                
                # Forward pass
                output = model(batch)
                
                # Consistent label handling
                batch_y = batch.y
                if batch_y.dim() > 2:  # Handle batch dimension issues
                    batch_y = batch_y.squeeze(0)
                
                # Debug shape issues
                # print(f"Output shape: {output.shape}, Target shape: {batch_y.shape}")
                
                # Calculate loss, backprop
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
                    
                    # Forward pass with sigmoid activation for binary classification
                    output = model(batch)
                    
                    # Consistent label handling - same as training
                    batch_y = batch.y
                    if batch_y.dim() > 2:
                        batch_y = batch_y.squeeze(0)
                    
                    # Calculate validation loss
                    loss = criterion(output, batch_y)
                    val_loss += loss.item()
                    
                    # Store predictions for metrics calculation
                    y_true.append(batch_y.cpu().numpy())
                    y_pred.append(torch.sigmoid(output).cpu().numpy())
            
            # Stack all batches for metrics calculation
            y_true = np.vstack(y_true)
            y_pred = np.vstack(y_pred)
            
            # Calculate metrics
            auc = roc_auc_score(y_true, y_pred, average='macro')
            y_pred_binary = (y_pred > 0.5).astype(int)
            precision = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
            
            # Average losses per batch
            train_loss = epoch_loss / len(train_loader)
            val_loss = val_loss / len(test_loader)
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            auc_scores.append(auc)
            
            # Print progress
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, "
                  f"AUC = {auc:.4f}, F1 = {f1:.4f}")
            
            # Early stopping
            if early_stopping.should_stop(val_loss):
                print("Early stopping triggered!")
                break
        
        # Store best metrics for this fold
        best_epoch = np.argmin(val_losses)
        fold_metrics['train_loss'].append(train_losses[best_epoch])
        fold_metrics['val_loss'].append(val_losses[best_epoch])
        fold_metrics['auc'].append(auc_scores[best_epoch])
        
        # Final evaluation on test set
        model.eval()
        all_y_true, all_y_pred = [], []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                output = torch.sigmoid(model(batch))
                
                batch_y = batch.y
                if batch_y.dim() > 2:
                    batch_y = batch_y.squeeze(0)
                
                all_y_true.append(batch_y.cpu().numpy())
                all_y_pred.append(output.cpu().numpy())
        
        all_y_true = np.vstack(all_y_true)
        all_y_pred = np.vstack(all_y_pred)
        all_y_pred_binary = (all_y_pred > 0.5).astype(int)
        
        # Calculate final metrics
        final_precision = precision_score(all_y_true, all_y_pred_binary, average='macro', zero_division=0)
        final_recall = recall_score(all_y_true, all_y_pred_binary, average='macro', zero_division=0)
        final_f1 = f1_score(all_y_true, all_y_pred_binary, average='macro', zero_division=0)
        
        fold_metrics['precision'].append(final_precision)
        fold_metrics['recall'].append(final_recall)
        fold_metrics['f1'].append(final_f1)
        
        print(f"Fold {fold+1} results - Precision: {final_precision:.4f}, Recall: {final_recall:.4f}, F1: {final_f1:.4f}")
        
        # Plot learning curves
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Fold {fold+1} - Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(auc_scores, label='AUC-ROC')
        plt.xlabel('Epochs')
        plt.ylabel('AUC')
        plt.legend()
        plt.title(f'Fold {fold+1} - AUC per Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(os.getenv('HOME'), f'Fold_{fold+1}_metrics_gsv3.png'))
        
        # Create confusion matrices
        y_pred_binary = (all_y_pred > 0.5).astype(int)
        conf_matrices = multilabel_confusion_matrix(all_y_true, y_pred_binary)
        num_classes = all_y_true.shape[1]
        
        fig, axes = plt.subplots(1, num_classes, figsize=(num_classes * 3, 4))
        if num_classes == 1:  # Handle the case of a single label
            axes = [axes]
            
        for i, (cm, ax) in enumerate(zip(conf_matrices, axes)):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"Class {i+1}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        
        plt.suptitle(f"Confusion Matrices - Fold {fold+1}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(os.getenv('HOME'), f'Fold_{fold+1}_confmat_gsv3.png'))
    
    # Print cross-validation summary
    print("\n===== Cross-Validation Summary =====")
    for metric, values in fold_metrics.items():
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    return fold_metrics


if __name__ == '__main__':
    # Load the dataset
    dataset_df = pd.read_csv('multi_label_dataset.csv')

    if 'smiles' not in dataset_df.columns:
        raise ValueError("The dataset must contain a 'smiles' column.")

    label_columns = [col for col in dataset_df.columns if col not in ['cid', 'smiles']]

    dataset = []
    for _, row in dataset_df.iterrows():
        smiles = row['smiles']
        labels = row[label_columns].tolist()
        graph = smiles_to_graph(smiles, labels)
        if graph is not None:
            dataset.append(graph)

    if not dataset:
        raise ValueError("No valid graphs were generated from the dataset.")

        # **Garantir que todos os objetos Data possuem global_features**
    for i, data in enumerate(dataset):
        if not hasattr(data, 'global_features'):
            print(f"Aviso: O dado {i} não tem global_features! Substituindo por zeros.")
            data.global_features = torch.zeros(5, dtype=torch.float)

    cross_validate(dataset, MODEL_HYPERPARAMS, OPTIMIZER_HYPERPARAMS)