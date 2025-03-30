import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
import copy
from typing import Dict, Any, List, Optional, Tuple
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast

def train_model(model: nn.Module, 
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               num_epochs: int = 100,
               learning_rate: float = 0.01,
               weight_decay: float = 5e-4,
               patience: int = 20,
               fp16: bool = True) -> Dict[str, Any]:
    """
    Train a model from scratch.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs to train
        learning_rate: Initial learning rate
        weight_decay: Weight decay (L2 regularization)
        patience: Number of epochs to wait for validation loss improvement
        fp16: Whether to use mixed precision training
        
    Returns:
        Dictionary with training history and best model state
    """
    logging.info(f"🏋️ Training model from scratch for {num_epochs} epochs")
    
    # Get device
    device = next(model.parameters()).device
    
    # Set model to training mode
    model.train()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Initialize mixed precision training if enabled
    scaler = GradScaler() if fp16 and torch.cuda.is_available() else None
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None
    
    # Initialize history dictionary
    history = {
        'loss': [],
        'accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(num_epochs):
        # Initialize metrics for this epoch
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Train for one epoch
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass with mixed precision if enabled
            if scaler is not None:
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward and backward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            
            # Update metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Compute epoch metrics
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        
        # Validate after each epoch
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Update history
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{num_epochs} - "
                    f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Check for improvement
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            logging.info(f"✅ New best model at epoch {epoch+1} with val_acc: {val_acc:.2f}%")
        else:
            epochs_no_improve += 1
            
        # Early stopping
        if epochs_no_improve >= patience:
            logging.info(f"⏹️ Early stopping at epoch {epoch+1} as validation accuracy did not improve for {patience} epochs")
            break
    
    # Calculate training time
    train_time = time.time() - start_time
    logging.info(f"✅ Training completed in {train_time:.2f} seconds")
    logging.info(f"🏆 Best model at epoch {best_epoch+1} with val_acc: {best_val_acc:.2f}%")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss,
        'training_time': train_time,
        'best_model_state': best_model_state
    }

def fine_tune_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, fp16=True):
    """
    Fine-tune a pruned model.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs
        learning_rate: Learning rate
        fp16: Whether to use mixed precision training
        
    Returns:
        dict: Results of fine-tuning
    """
    logging.info(f"🔄 Fine-tuning model for {num_epochs} epochs with lr={learning_rate}")
    
    # Set model to training mode
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Use mixed precision training if available
    scaler = GradScaler(enabled=fp16 and torch.cuda.is_available())
    
    # Initialize variables for tracking best model
    best_val_acc = 0.0
    best_epoch = 0
    best_state_dict = copy.deepcopy(model.state_dict())
    
    # Training history
    history = {
        'loss': [],
        'acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Start training timer
    start_time = time.time()
    
    # Start fine-tuning
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Mixed precision training
            if scaler.is_enabled():
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Scale loss and backpropagate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular training
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader)
        
        # Update history
        history['loss'].append(train_loss)
        history['acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log progress
        logging.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            logging.info(f"✅ New best model at epoch {epoch+1} with val_acc: {val_acc:.2f}%")
    
    # Calculate training time
    training_time = time.time() - start_time
    logging.info(f"✅ Fine-tuning completed in {training_time:.2f} seconds")
    
    # Restore best model
    model.load_state_dict(best_state_dict)
    logging.info(f"🏆 Best model at epoch {best_epoch+1} with val_acc: {best_val_acc:.2f}%")
    
    # Return results
    return {
        'history': history,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc,
        'training_time': training_time
    }

def evaluate_model(model: nn.Module, 
                  data_loader: torch.utils.data.DataLoader,
                  criterion: Optional[nn.Module] = None) -> Tuple[float, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader for evaluation
        criterion: Loss function (optional)
        
    Returns:
        Tuple of (loss, accuracy)
    """
    # Get device
    device = next(model.parameters()).device
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Use cross entropy loss if no criterion is provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Evaluate the model
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    # Calculate metrics
    loss = running_loss / total if total > 0 else 0.0
    accuracy = 100 * correct / total if total > 0 else 0.0
    
    # Restore model to training mode
    model.train()
    
    return loss, accuracy 