from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from merge_batchnorm import *
from units import *
from models import *
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import json
from datetime import datetime
import seaborn as sns
from collections import defaultdict
import pandas as pd
import csv


def train(model, device, train_loader, optimizer, epoch):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        onehot = torch.nn.functional.one_hot(target, 20).float()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        # Print less frequently to reduce output
        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')
    
    return total_loss / len(train_loader)

def validate(model, device, val_loader, testdataset_=False, return_cm=False):
    model.eval()
    correct = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=False)
            correct += pred.eq(target.view_as(pred)).sum().item()
            all_preds.extend(pred.view(-1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * correct / len(val_loader.dataset)
    cm = confusion_matrix(all_targets, all_preds)
    
    print(f'Accuracy: {accuracy:.2f}%')
    
    if return_cm:
        return accuracy, cm, all_preds, all_targets
    elif testdataset_:
        return accuracy
    else:
        return accuracy

def single_run(run_id, args, device):
    """
    Perform a single training run and return results
    """
    print(f"\n{'='*60}")
    print(f"RUN {run_id + 1}/5")
    print(f"{'='*60}")
    
    # Set different seed for each run
    torch.manual_seed(3407+17+run_id)
    np.random.seed(3407+17+run_id)
    
    # Load data
    train_data = torch.load('train_data_3dimension_force.pt')
    val_data = torch.load('val_data_3dimension_force.pt')
    test_data = torch.load('test_data_3dimension_force.pt')

    # train_data 300
    # val
    # test_data
    train_loader = DataLoader(TensorDataset(train_data['X'], train_data['y']), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data['X'], val_data['y']), batch_size=2000, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_data['X'], test_data['y']), batch_size=2000, shuffle=False)

    # Model configuration
    input_size = 450
    hidden_sizes = [384, 512, 768, 128]
    output_size = 20
    custom_dropouts = [0.10, 0.10, 0.35, 0.25]
    # Create model
    model = Net(input_size=input_size, hidden_sizes=hidden_sizes, output_size=output_size,
                quantize_level=args.T, dropout_rates=custom_dropouts).to(device)
    
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        
    # Training tracking
    best_val_acc = 0.0
    best_model_path = f"snn_model_run_{run_id}.pt"
    train_losses = []
    val_accuracies = []
    test_accuracies = []
    
    print(f"Starting training for run {run_id + 1}...")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        if epoch % 10 == 0:  # Print every 10 epochs
            print(f"Run {run_id + 1} - Epoch {epoch}/{args.epochs}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_loss = train(model, device, train_loader, optimizer, epoch)
        train_losses.append(train_loss)
        
        # Validate
        val_acc = validate(model, device, val_loader, testdataset_=True)
        test_acc = validate(model, device, test_loader, testdataset_=True)
        
        val_accuracies.append(val_acc)
        test_accuracies.append(test_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            #if epoch % 10 == 0:
            print(f"Run {run_id + 1} - New best model saved with validation accuracy: {val_acc:.2f}%")
        
        scheduler.step()
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    print(f"\nRun {run_id + 1} - Final ANN evaluation:")
    final_ann_acc, ann_cm, ann_preds, ann_targets = validate(model, device, test_loader, testdataset_=True, return_cm=True)
    
    # Convert to quantized model
    print(f"Run {run_id + 1} - Converting to quantized model...")
    model_inference = InferenceNet(input_size=input_size, hidden_sizes=hidden_sizes, 
                                 output_size=output_size, quantize_level=args.T).to(device)
    model_inference = transfer_weights_to_inference_model(model, model_inference)
    
    # Convert to SNN
    print(f"Run {run_id + 1} - Converting to SNN...")
    snn_quantized = Sparrow_SNN(input_size=input_size, hidden_sizes=hidden_sizes, 
                               output_size=output_size, quantized_index=8, T=args.T, 
                               Hybrid=args.Hybrid).to(device)
    snn_quantized.load_state_dict(model_inference.state_dict(), strict=False)
    
    # Final SNN evaluation
    print(f"Run {run_id + 1} - Final SNN evaluation:")
    final_snn_acc, snn_cm, snn_preds, snn_targets = validate(snn_quantized, device, test_loader, testdataset_=True, return_cm=True)
    
    # Clean up model file
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    
    return {
        'run_id': run_id + 1,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'test_accuracies': test_accuracies,
        'best_val_acc': best_val_acc,
        'final_ann_acc': final_ann_acc,
        'final_snn_acc': final_snn_acc,
        'ann_confusion_matrix': ann_cm,
        'snn_confusion_matrix': snn_cm,
        'ann_predictions': ann_preds,
        'ann_targets': ann_targets,
        'snn_predictions': snn_preds,
        'snn_targets': snn_targets
    }

def plot_results(all_results, args):
    """
    Create simplified visualizations focusing on SNN confusion matrix and validation accuracy evolution
    """
    print("\nCreating simplified SNN visualizations...")
    
    # Create results directory
    results_dir = "results_450d_onlyforce"
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}/")
    
    # Set publication-quality style
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
        'legend.frameon': True,
        'legend.fancybox': True,
        'legend.shadow': True,
        'legend.fontsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    
    # Create figure with 2 subplots
    fig = plt.figure(figsize=(16, 6))
    
    # ============ SUBPLOT 1: Average SNN Confusion Matrix ============
    ax1 = plt.subplot(1, 2, 1)
    
    # Calculate average confusion matrix across all runs
    avg_snn_cm = np.mean([result['snn_confusion_matrix'] for result in all_results], axis=0)
    
    # Normalize to percentages
    avg_snn_cm_norm = avg_snn_cm / avg_snn_cm.sum(axis=1, keepdims=True) * 100
    
    # Create heatmap
    im = plt.imshow(avg_snn_cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    
    # Add colorbar
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label('Classification Accuracy (%)', fontweight='bold')
    
    # Add text annotations for all elements
    for i in range(avg_snn_cm_norm.shape[0]):
        for j in range(avg_snn_cm_norm.shape[1]):
            if avg_snn_cm_norm[i, j] > 50:  # White text for dark cells
                color = 'white'
            else:  # Black text for light cells
                color = 'black'
            
            text = plt.text(j, i, f'{avg_snn_cm_norm[i, j]:.1f}', 
                          ha="center", va="center", color=color, 
                          fontsize=8, fontweight='bold')
    
    # Set labels and title
    plt.xlabel('Predicted Class', fontweight='bold')
    plt.ylabel('True Class', fontweight='bold')
    #plt.title('Average SNN Confusion Matrix (450D, 5 Runs)', fontweight='bold', pad=15)
    
    # Set tick labels
    num_classes = avg_snn_cm_norm.shape[0]
    plt.xticks(range(num_classes), range(num_classes))
    plt.yticks(range(num_classes), range(num_classes))
    
    # ============ SUBPLOT 2: Validation Accuracy Evolution Over 310 Epochs ============
    ax2 = plt.subplot(1, 2, 2)
    
    # Professional color palette for different runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot validation accuracy for each run
    for i, result in enumerate(all_results):
        epochs = range(1, len(result['val_accuracies']) + 1)
        plt.plot(epochs, result['val_accuracies'], color=colors[i % len(colors)], 
                alpha=0.7, linewidth=1.5, label=f'Run {i+1}')
    
    plt.xlabel('Training Epoch', fontweight='bold')
    plt.ylabel('Validation Accuracy (%)', fontweight='bold')
    plt.legend(loc='lower right', ncol=2, fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    # Save figure to results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = os.path.join(results_dir, f'snn_450d_results_{timestamp}.pdf')
    plt.savefig(plot_filename, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(plot_filename.replace('.pdf', '.png'), format='png', dpi=300, bbox_inches='tight')
    
    print(f"450D SNN results figure saved as: {plot_filename}")
    
    # Calculate overall SNN accuracy from confusion matrix
    overall_accuracy = np.trace(avg_snn_cm_norm) / avg_snn_cm_norm.shape[0]
    print(f"\nAverage SNN Classification Accuracy: {overall_accuracy:.2f}%")
    
    # Show plot
    plt.show()
    
    # Reset matplotlib parameters
    plt.rcdefaults()
    
    return plot_filename

def save_training_data_to_csv(all_results, results_dir, timestamp):
    """
    Save training process data (validation accuracies, test accuracies, losses) to CSV files
    """
    print("Saving training process data to CSV files...")
    
    # 1. Save individual run training curves
    for i, result in enumerate(all_results):
        run_data = {
            'epoch': list(range(1, len(result['val_accuracies']) + 1)),
            'train_loss': result['train_losses'],
            'val_accuracy': result['val_accuracies'],
            'test_accuracy': result['test_accuracies']
        }
        
        df = pd.DataFrame(run_data)
        csv_file = os.path.join(results_dir, f'training_curves_450d_run_{i+1}_{timestamp}.csv')
        df.to_csv(csv_file, index=False)
        print(f"  Run {i+1} training curves saved: {csv_file}")
    
    # 2. Save combined validation accuracies, test accuracies, and losses across all runs
    max_epochs = max(len(result['val_accuracies']) for result in all_results)
    combined_data = {'epoch': list(range(1, max_epochs + 1))}
    
    for i, result in enumerate(all_results):
        # Pad with NaN if some runs have fewer epochs
        val_accs = result['val_accuracies'] + [None] * (max_epochs - len(result['val_accuracies']))
        test_accs = result['test_accuracies'] + [None] * (max_epochs - len(result['test_accuracies']))
        train_losses = result['train_losses'] + [None] * (max_epochs - len(result['train_losses']))
        
        combined_data[f'val_accuracy_run_{i+1}'] = val_accs
        combined_data[f'test_accuracy_run_{i+1}'] = test_accs
        combined_data[f'train_loss_run_{i+1}'] = train_losses
    
    combined_df = pd.DataFrame(combined_data)
    combined_csv = os.path.join(results_dir, f'all_runs_training_curves_450d_{timestamp}.csv')
    combined_df.to_csv(combined_csv, index=False)
    print(f"  Combined training curves saved: {combined_csv}")
    
    return combined_csv

def save_confusion_matrices_to_csv(all_results, results_dir, timestamp):
    """
    Save SNN confusion matrices to CSV files
    """
    print("Saving SNN confusion matrices to CSV files...")
    
    # 1. Save individual SNN confusion matrices
    for i, result in enumerate(all_results):
        # SNN confusion matrix
        snn_cm_df = pd.DataFrame(result['snn_confusion_matrix'],
                                index=[f'True_{j}' for j in range(20)], 
                                columns=[f'Pred_{j}' for j in range(20)])
        snn_csv = os.path.join(results_dir, f'snn_confusion_matrix_450d_run_{i+1}_{timestamp}.csv')
        snn_cm_df.to_csv(snn_csv)
        print(f"  Run {i+1} SNN confusion matrix saved: {snn_csv}")
    
    # 2. Save average SNN confusion matrix
    avg_snn_cm = np.mean([result['snn_confusion_matrix'] for result in all_results], axis=0)
    
    # Average SNN confusion matrix
    avg_snn_df = pd.DataFrame(avg_snn_cm,
                             index=[f'True_{j}' for j in range(20)],
                             columns=[f'Pred_{j}' for j in range(20)])
    avg_snn_csv = os.path.join(results_dir, f'avg_snn_confusion_matrix_450d_{timestamp}.csv')
    avg_snn_df.to_csv(avg_snn_csv)
    print(f"  Average SNN confusion matrix saved: {avg_snn_csv}")
    
    # 3. Save normalized SNN confusion matrix (percentages)
    avg_snn_cm_norm = avg_snn_cm / avg_snn_cm.sum(axis=1, keepdims=True) * 100
    
    avg_snn_norm_df = pd.DataFrame(avg_snn_cm_norm,
                                  index=[f'True_{j}' for j in range(20)],
                                  columns=[f'Pred_{j}' for j in range(20)])
    avg_snn_norm_csv = os.path.join(results_dir, f'avg_snn_confusion_matrix_normalized_450d_{timestamp}.csv')
    avg_snn_norm_df.to_csv(avg_snn_norm_csv)
    print(f"  Average SNN confusion matrix (normalized) saved: {avg_snn_norm_csv}")
    
    return avg_snn_csv, avg_snn_norm_csv

def save_summary_results_to_csv(all_results, results_dir, timestamp):
    """
    Save summary results for each run to CSV
    """
    print("Saving summary results to CSV...")
    
    summary_data = {
        'run_id': [],
        'best_val_accuracy': [],
        'final_ann_accuracy': [],
        'final_snn_accuracy': [],
        'accuracy_drop': []
    }
    
    for result in all_results:
        summary_data['run_id'].append(result['run_id'])
        summary_data['best_val_accuracy'].append(result['best_val_acc'])
        summary_data['final_ann_accuracy'].append(result['final_ann_acc'])
        summary_data['final_snn_accuracy'].append(result['final_snn_acc'])
        summary_data['accuracy_drop'].append(result['final_ann_acc'] - result['final_snn_acc'])
    
    # Add statistics
    ann_accuracies = summary_data['final_ann_accuracy']
    snn_accuracies = summary_data['final_snn_accuracy']
    accuracy_drops = summary_data['accuracy_drop']
    
    # Add summary row
    summary_data['run_id'].append('MEAN')
    summary_data['best_val_accuracy'].append(np.mean([result['best_val_acc'] for result in all_results]))
    summary_data['final_ann_accuracy'].append(np.mean(ann_accuracies))
    summary_data['final_snn_accuracy'].append(np.mean(snn_accuracies))
    summary_data['accuracy_drop'].append(np.mean(accuracy_drops))
    
    summary_data['run_id'].append('STD')
    summary_data['best_val_accuracy'].append(np.std([result['best_val_acc'] for result in all_results]))
    summary_data['final_ann_accuracy'].append(np.std(ann_accuracies))
    summary_data['final_snn_accuracy'].append(np.std(snn_accuracies))
    summary_data['accuracy_drop'].append(np.std(accuracy_drops))
    
    summary_df = pd.DataFrame(summary_data)
    summary_csv = os.path.join(results_dir, f'summary_results_450d_onlyforce_{timestamp}.csv')
    summary_df.to_csv(summary_csv, index=False)
    print(f"  Summary results saved: {summary_csv}")
    
    return summary_csv

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='SNN Multi-Run Classification')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=310, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=2e-2, metavar='LR',
                        help='learning rate (default: 2e-2)')
    parser.add_argument('--T', type=int, default=31, metavar='LR',
                        help='time window size')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--Hybrid', type=str, default=False, metavar='RESUME',
                        help='Resume model from checkpoint')
    parser.add_argument('--num-runs', type=int, default=5, metavar='N',
                        help='number of runs to perform (default: 5)')
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    print(f"Device: {device}")
    print(f"Running {args.num_runs} independent training runs...")
    print(f"Configuration: {args.epochs} epochs, LR={args.lr}, T={args.T}")
    
    # Store results from all runs
    all_results = []
    
    # Run multiple training sessions
    for run_id in range(args.num_runs):
        try:
            result = single_run(run_id, args, device)
            all_results.append(result)
            print(f"Run {run_id + 1} completed successfully!")
            print(f"  ANN Accuracy: {result['final_ann_acc']:.2f}%")
            print(f"  SNN Accuracy: {result['final_snn_acc']:.2f}%")
        except Exception as e:
            print(f"Run {run_id + 1} failed with error: {e}")
            continue
    
    if not all_results:
        print("No successful runs completed!")
        return
    
    # Calculate statistics
    print(f"\n{'='*60}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'='*60}")
    
    ann_accuracies = [result['final_ann_acc'] for result in all_results]
    snn_accuracies = [result['final_snn_acc'] for result in all_results]
    
    print(f"\nANN Test Accuracies: {[f'{acc:.2f}%' for acc in ann_accuracies]}")
    print(f"SNN Test Accuracies: {[f'{acc:.2f}%' for acc in snn_accuracies]}")
    
    # ANN Statistics
    ann_mean = np.mean(ann_accuracies)
    ann_std = np.std(ann_accuracies)
    ann_ci = 1.96 * ann_std / np.sqrt(len(ann_accuracies))  # 95% confidence interval
    
    print(f"\nANN Results:")
    print(f"  Mean ± Std: {ann_mean:.2f}% ± {ann_std:.2f}%")
    print(f"  95% CI: [{ann_mean - ann_ci:.2f}%, {ann_mean + ann_ci:.2f}%]")
    print(f"  Range: [{min(ann_accuracies):.2f}%, {max(ann_accuracies):.2f}%]")
    
    # SNN Statistics
    snn_mean = np.mean(snn_accuracies)
    snn_std = np.std(snn_accuracies)
    snn_ci = 1.96 * snn_std / np.sqrt(len(snn_accuracies))
    
    print(f"\nSNN Results:")
    print(f"  Mean ± Std: {snn_mean:.2f}% ± {snn_std:.2f}%")
    print(f"  95% CI: [{snn_mean - snn_ci:.2f}%, {snn_mean + snn_ci:.2f}%]")
    print(f"  Range: [{min(snn_accuracies):.2f}%, {max(snn_accuracies):.2f}%]")
    
    # Accuracy drop analysis
    accuracy_drops = [ann - snn for ann, snn in zip(ann_accuracies, snn_accuracies)]
    drop_mean = np.mean(accuracy_drops)
    drop_std = np.std(accuracy_drops)
    
    print(f"\nAccuracy Drop (ANN → SNN):")
    print(f"  Mean ± Std: {drop_mean:.2f}% ± {drop_std:.2f}%")
    print(f"  Range: [{min(accuracy_drops):.2f}%, {max(accuracy_drops):.2f}%]")
    
    # Target result
    print(f"\n🎯 AVERAGE SNN TEST ACCURACY: {snn_mean:.2f}%")
    
    # Create results directory
    results_dir = "results_450d_onlyforce"
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results to the results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'snn_450d_results_{timestamp}.json')
    
    # Prepare data for JSON serialization
    save_data = {
        'timestamp': timestamp,
        'configuration': {
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'time_window': args.T,
            'num_runs': len(all_results),
            'input_size': 450,  # Original 450D input
            'architecture': [384, 512, 768, 128],  # Correct hidden layer sizes
            'dropout_rates': [0.10, 0.10, 0.35, 0.25] # Correct dropout rates
        },
        'results': {
            'ann_accuracies': ann_accuracies,
            'snn_accuracies': snn_accuracies,
            'accuracy_drops': accuracy_drops
        },
        'statistics': {
            'ann': {
                'mean': float(ann_mean),
                'std': float(ann_std),
                'ci_95': [float(ann_mean - ann_ci), float(ann_mean + ann_ci)],
                'range': [float(min(ann_accuracies)), float(max(ann_accuracies))]
            },
            'snn': {
                'mean': float(snn_mean),
                'std': float(snn_std),
                'ci_95': [float(snn_mean - snn_ci), float(snn_mean + snn_ci)],
                'range': [float(min(snn_accuracies)), float(max(snn_accuracies))]
            },
            'accuracy_drop': {
                'mean': float(drop_mean),
                'std': float(drop_std),
                'range': [float(min(accuracy_drops)), float(max(accuracy_drops))]
            }
        },
        'all_results': [
            {
                'run_id': result['run_id'],
                'final_snn_acc': result['final_snn_acc'],
                'snn_confusion_matrix': result['snn_confusion_matrix'].tolist()
            }
            for result in all_results
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    # Create comprehensive visualizations
    plot_filename = plot_results(all_results, args)
    
    print(f"\n{'='*60}")
    print("450D MULTI-RUN ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"📁 Results directory: {results_dir}/")
    print(f"📊 Results file: {results_file}")
    print(f"📈 Main figure: {plot_filename}")
    print(f"🎯 Average SNN accuracy: {snn_mean:.2f}% ± {snn_std:.2f}%")
    print(f"📋 Input dimension: 450D (original)")
    print(f"🔄 Number of runs: {len(all_results)}")
    print(f"⚡ Training epochs: {args.epochs}")
    
    # List all files in results directory
    import glob
    result_files = glob.glob(os.path.join(results_dir, "*"))
    print(f"\n📂 Files in {results_dir}/:")
    for file in sorted(result_files):
        file_size = os.path.getsize(file) / 1024  # KB
        print(f"   {os.path.basename(file)} ({file_size:.1f} KB)")

    #Save training data to CSV
    save_training_data_to_csv(all_results, results_dir, timestamp)

    # Save confusion matrices to CSV
    save_confusion_matrices_to_csv(all_results, results_dir, timestamp)

    # Save summary results to CSV
    save_summary_results_to_csv(all_results, results_dir, timestamp)

if __name__ == '__main__':
    main() 