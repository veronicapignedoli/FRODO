#!/usr/bin/env python3
"""
PARAMETER SWEEP SCRIPT FOR 3D RIM CLASSIFIER

This script performs a systematic grid search over multiple hyperparameters
to find the best configuration for the 3D Rim Classifier.

USAGE:
    python parameter_sweep.py

PARAMETERS TESTED:
    - dropout: Dropout rate (e.g., 0.3, 0.5, 0.7)
    - modalities: MRI modalities used (e.g., ['T1', 'QSM', 'QSMp'])
    - filters: Number of base filters (e.g., 16, 32, 64)
    - batch_size: Training batch size (e.g., 10, 15, 20)
    - patience: Early stopping patience (e.g., 10, 15, 20)
    - learning_rate: Optimizer learning rate (e.g., 1e-5, 5e-5, 1e-4)
    - threshold: Classification threshold (e.g., 0.4, 0.5, 0.6)

OUTPUT:
    - sweep_results_<timestamp>/
        ├── sweep_log.txt           # Detailed log of all experiments
        ├── results_summary.json    # JSON with all results
        └── final_summary.txt       # Top 5 models by F1, Recall, and AUC

CONFIGURATION:
    Edit the param_grid dictionary below to customize the search space.
    For quick testing, use the commented example with single values.

MONITORING:
    Watch progress in real-time:
        tail -f sweep_results_*/sweep_log.txt
"""
import os
import sys
import subprocess
import itertools
from itertools import combinations
from datetime import datetime
import json
import time
import re

def create_sweep_script():
    """Crea lo script per il parameter sweep"""
    
    # DEFINISCI I PARAMETRI DA TESTARE
    param_grid = {
        'dropout': [ 0.5],
        'modalities': [['T1', 'QSM', 'QSMp'], ['T1', 'QSM'], ['QSM', 'QSMp']],
        'filters': [ 32],
        'batch_size': [10, 15],
        'patience': [10],
        'learning_rate': [1e-5, 5e-5, 1e-4],
        'threshold': [0.3]
    }

    """
    Esempio di configurazione ridotta per test:
    param_grid = {
        'dropout': [0.5],
        'modalities': [['T1', 'QSM', 'QSMp']],
        'filters': [32],
        'batch_size': [15],
        'patience': [10],
        'learning_rate': [5e-5],
        'threshold': [0.5]
    }
    """
    
    # CONFIGURAZIONI FISSE
    fixed_params = {
        'dataset_type': 'HSMn',
        'epochs': 100,
        'gpu': 0,
        'seed': 42
    }
    
    # CREA DIRECTORY RESULTS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"sweep_results_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # FILE DI LOG PRINCIPALE
    log_file = os.path.join(results_dir, "sweep_log.txt")
    results_file = os.path.join(results_dir, "results_summary.json")
    
    print(f"🚀 Starting parameter sweep at {datetime.now()}")
    print(f"📁 Results will be saved to: {results_dir}")
    print(f"📊 Parameter grid: {param_grid}")
    
    # GENERA TUTTE LE COMBINAZIONI
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))
    
    print(f"🔢 Total combinations to test: {len(combinations)}")
    
    all_results = []
    
    with open(log_file, 'w') as f:
        f.write(f"PARAMETER SWEEP LOG - Started at {datetime.now()}\n")
        f.write("="*80 + "\n\n")
    
    for i, combo in enumerate(combinations):
        print(f"\n{'='*60}")
        print(f"🧪 EXPERIMENT {i+1}/{len(combinations)}")
        print(f"{'='*60}")
        
        # CREA DIZIONARIO PARAMETRI PER QUESTA COMBINAZIONE
        current_params = dict(zip(param_names, combo))
        current_params.update(fixed_params)
        
        # NOME EXPERIMENT UNICO
        exp_name = f"rim_classifier_{i+1}"
        current_params['experiment'] = exp_name
        
        print(f"📋 Parameters: {current_params}")
        
        # ESEGUI IL TRAINING
        try:
            result = run_single_experiment(current_params, results_dir, log_file, i+1, len(combinations))
            result['experiment_id'] = i + 1
            result['experiment_name'] = exp_name
            result['parameters'] = current_params
            all_results.append(result)
            
            print(f"✅ Experiment {i+1} ({exp_name}) completed successfully!")
            if 'accuracy' in result:
                print(f"📊 Accuracy: {result['accuracy']:.4f}")
            if 'f1' in result:
                print(f"📊 F1-Score: {result['f1']:.4f}")
            
        except Exception as e:
            error_msg = f"❌ Experiment {i+1} ({exp_name}) failed: {str(e)}"
            print(error_msg)
            
            # LOG ERRORE
            with open(log_file, 'a') as f:
                f.write(f"EXPERIMENT {i+1}/{len(combinations)} - {exp_name}\n")
                f.write(f"Started at: {datetime.now()}\n")
                f.write(f"ERROR: {error_msg}\n")
                f.write("="*80 + "\n\n")
            
            # Aggiungi risultato con errore
            all_results.append({
                'experiment_id': i + 1,
                'experiment_name': exp_name,
                'parameters': current_params,
                'status': 'failed',
                'error': str(e)
            })
        
        # SALVA RISULTATI INTERMEDI
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
    
    # CREA SUMMARY FINALE
    create_final_summary(all_results, results_dir)
    
    print(f"\n🎉 Parameter sweep completed!")
    print(f"📁 Results saved in: {results_dir}")
    print(f"📊 Check {results_file} for detailed results")

def run_single_experiment(params, results_dir, log_file, exp_number, total_combinations):
    """Esegue un singolo esperimento con i parametri dati"""
    
    # COSTRUISCI IL COMANDO
    cmd = ['python', 'main.py']
    
    # AGGIUNGI PARAMETRI
    for key, value in params.items():
        if key == 'dataset_type':
            cmd.extend(['-dt', str(value)])
        elif key == 'epochs':
            cmd.extend(['-ep', str(value)])
        elif key == 'dropout':
            cmd.extend(['-drop', str(value)])
        elif key == 'modalities':
            # Le modalità sono una lista, le passiamo come stringa separata da virgole
            mod_str = ','.join(value)
            cmd.extend(['-mod', mod_str])
        elif key == 'filters':
            cmd.extend(['-f', str(value)])
        elif key == 'experiment':
            cmd.extend(['-exp', str(value)])
        elif key == 'batch_size':
            cmd.extend(['-batch', str(value)])
        elif key == 'patience':
            cmd.extend(['-patience', str(value)])
        elif key == 'learning_rate':
            cmd.extend(['-lr', str(value)])
        elif key == 'threshold':
            cmd.extend(['-th', str(value)])
        elif key == 'gpu':
            cmd.extend(['-g', str(value)])
        elif key == 'seed':
            cmd.extend(['-seed', str(value)])
    
    print(f"🔄 Running command: {' '.join(cmd)}")
    
    # ESEGUI IL COMANDO E CATTURA OUTPUT
    start_time = time.time()
    
    # CATTURA OUTPUT SENZA SALVARE FILE SEPARATI
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    output_lines = []
    # CATTURA OUTPUT IN TEMPO REALE
    for line in process.stdout:
        output_lines.append(line)
        
        # MOSTRA OUTPUT IMPORTANTE
        if any(keyword in line.lower() for keyword in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'error', 'epoch']):
            print(f"📝 {line.strip()}")
    
    process.wait()
    
    end_time = time.time()
    duration = end_time - start_time
    
    if process.returncode != 0:
        raise Exception(f"Process failed with return code {process.returncode}")
    
    # UNISCI TUTTO L'OUTPUT IN UN SINGOLO STRING
    full_output = ''.join(output_lines)
    
    # ESTRAI INFORMAZIONI DETTAGLIATE
    experiment_info = extract_experiment_info(full_output, params, duration)
    
    # SCRIVI NEL LOG PRINCIPALE CON IL FORMATO RICHIESTO
    write_experiment_log(log_file, exp_number, params['experiment'], experiment_info, total_combinations, params)
    
    return experiment_info

def extract_experiment_info(output, params, duration):
    """Estrae tutte le informazioni richieste dall'output"""
    info = {
        'status': 'completed',
        'duration_seconds': duration
    }
    
    # ESTRAI METRICHE DI TEST (dalla funzione evaluate_classifier)
    metrics_patterns = {
        'accuracy': r'Accuracy:\s*([\d.]+)',
        'precision': r'Precision:\s*([\d.]+)',
        'recall': r'Recall:\s*([\d.]+)',
        'f1': r'F1-score:\s*([\d.]+)',
        'auc': r'Auc-roc:\s*([\d.]+)'
    }
    
    for metric, pattern in metrics_patterns.items():
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            info[metric] = float(match.group(1))
    
    # ESTRAI TRAINING TIME
    time_match = re.search(r'Training time:\s*([^\n]+)', output)
    if time_match:
        info['training_time_formatted'] = time_match.group(1).strip()
    
    avg_epoch_match = re.search(r'Average per epoch:\s*([\d.]+)s', output)
    if avg_epoch_match:
        info['avg_per_epoch'] = float(avg_epoch_match.group(1))
    
    # ESTRAI LABEL DISTRIBUTION
    rim_match = re.search(r'Rim \(1\):\s*(\d+)\s*\(([\d.]+)%\)', output)
    if rim_match:
        info['rim_samples'] = int(rim_match.group(1))
        info['rim_percentage'] = float(rim_match.group(2))
    
    norim_match = re.search(r'NoRim \(0\):\s*(\d+)\s*\(([\d.]+)%\)', output)
    if norim_match:
        info['norim_samples'] = int(norim_match.group(1))
        info['norim_percentage'] = float(norim_match.group(2))
    
    # ESTRAI DATASET SIZE
    train_match = re.search(r'Train:\s*(\d+)\s*patches', output)
    if train_match:
        info['train_patches'] = int(train_match.group(1))
    
    val_match = re.search(r'Val:\s*(\d+)\s*patches', output)
    if val_match:
        info['val_patches'] = int(val_match.group(1))
    
    test_match = re.search(r'Test:\s*(\d+)\s*patches', output)
    if test_match:
        info['test_patches'] = int(test_match.group(1))
    
    return info

def write_experiment_log(log_file, exp_number, exp_name, info, total_combinations, params):
    """Scrive il log dell'esperimento nel formato richiesto"""
    
    with open(log_file, 'a') as f:
        f.write(f"EXPERIMENT {exp_number}/{total_combinations} - {exp_name}\n")
        f.write(f"Started at: {datetime.now()}\n")
        f.write("-" * 60 + "\n")
        
        # METRICHE DI TEST
        if 'accuracy' in info:
            f.write(f"Accuracy: {info['accuracy']:.4f}\n")
        if 'precision' in info:
            f.write(f"Precision: {info['precision']:.4f}\n")
        if 'recall' in info:
            f.write(f"Recall: {info['recall']:.4f}\n")
        if 'f1' in info:
            f.write(f"F1-Score: {info['f1']:.4f}\n")
        if 'auc' in info:
            f.write(f"AUC-ROC: {info['auc']:.4f}\n")
        
        # DATASET INFO
        if 'train_patches' in info:
            f.write(f"\nDataset Size:\n")
            f.write(f"  Train: {info['train_patches']} patches\n")
            if 'val_patches' in info:
                f.write(f"  Val: {info['val_patches']} patches\n")
            if 'test_patches' in info:
                f.write(f"  Test: {info['test_patches']} patches\n")
        
        # LABEL DISTRIBUTION
        if 'rim_samples' in info:
            f.write(f"\nLabel Distribution:\n")
            f.write(f"  Rim: {info['rim_samples']} ({info['rim_percentage']:.1f}%)\n")
            f.write(f"  NoRim: {info['norim_samples']} ({info['norim_percentage']:.1f}%)\n")
        
        # TRAINING TIME
        if 'training_time_formatted' in info:
            f.write(f"\nTraining Time: {info['training_time_formatted']}\n")
            if 'avg_per_epoch' in info:
                f.write(f"Average per epoch: {info['avg_per_epoch']:.1f}s\n")
        
        # PARAMETRI
        f.write(f"\nParameters:\n")
        f.write(f"  Dropout: {params.get('dropout', 'N/A')}\n")
        f.write(f"  Modalities: {params.get('modalities', 'N/A')}\n")
        f.write(f"  Filters: {params.get('filters', 'N/A')}\n")
        f.write(f"  Batch size: {params.get('batch_size', 'N/A')}\n")
        f.write(f"  Patience: {params.get('patience', 'N/A')}\n")
        f.write(f"  Learning rate: {params.get('learning_rate', 'N/A')}\n")
        f.write(f"  Threshold: {params.get('threshold', 'N/A')}\n")
        
        f.write("="*80 + "\n\n")


def create_final_summary(all_results, results_dir):
    """Crea un summary finale con i migliori risultati"""
    
    summary_file = os.path.join(results_dir, "final_summary.txt")
    
    # FILTRA RISULTATI RIUSCITI
    successful_results = [r for r in all_results if r.get('status') == 'completed' and 'f1' in r]
    
    with open(summary_file, 'w') as f:
        f.write("PARAMETER SWEEP FINAL SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total experiments: {len(all_results)}\n")
        f.write(f"Successful: {len(successful_results)}\n")
        f.write(f"Failed: {len(all_results) - len(successful_results)}\n\n")
        
        if successful_results:
            # TOP 5 PER F1-SCORE
            f.write("🏆 TOP 5 RESULTS (by F1-Score):\n")
            f.write("-" * 80 + "\n")
            
            best_f1 = sorted(successful_results, key=lambda x: x['f1'], reverse=True)
            for i, result in enumerate(best_f1[:5]):
                f.write(f"\n{i+1}. {result['experiment_name']}\n")
                f.write(f"   F1-Score: {result['f1']:.4f}\n")
                if 'accuracy' in result:
                    f.write(f"   Accuracy: {result['accuracy']:.4f}\n")
                if 'precision' in result:
                    f.write(f"   Precision: {result['precision']:.4f}\n")
                if 'recall' in result:
                    f.write(f"   Recall: {result['recall']:.4f}\n")
                if 'auc' in result:
                    f.write(f"   AUC-ROC: {result['auc']:.4f}\n")
                
                f.write(f"   Parameters:\n")
                for param in ['dropout', 'modalities', 'filters', 'batch_size', 'patience', 'learning_rate', 'threshold']:
                    if param in result['parameters']:
                        f.write(f"     {param}: {result['parameters'][param]}\n")
            
            # TOP 5 PER RECALL
            f.write("\n\n🏆 TOP 5 RESULTS (by Recall):\n")
            f.write("-" * 80 + "\n")
            
            best_recall = sorted(successful_results, key=lambda x: x.get('recall', 0), reverse=True)
            for i, result in enumerate(best_recall[:5]):
                f.write(f"\n{i+1}. {result['experiment_name']}\n")
                f.write(f"   Recall: {result.get('recall', 'N/A'):.4f}\n")
                f.write(f"   F1-Score: {result['f1']:.4f}\n")
                if 'precision' in result:
                    f.write(f"   Precision: {result['precision']:.4f}\n")
                
                f.write(f"   Parameters:\n")
                for param in ['dropout', 'modalities', 'filters', 'batch_size', 'patience', 'learning_rate', 'threshold']:
                    if param in result['parameters']:
                        f.write(f"     {param}: {result['parameters'][param]}\n")
            
            # TOP 5 PER AUC
            if any('auc' in r for r in successful_results):
                f.write("\n\n🏆 TOP 5 RESULTS (by AUC-ROC):\n")
                f.write("-" * 80 + "\n")
                
                best_auc = sorted([r for r in successful_results if 'auc' in r], 
                                key=lambda x: x['auc'], reverse=True)
                for i, result in enumerate(best_auc[:5]):
                    f.write(f"\n{i+1}. {result['experiment_name']}\n")
                    f.write(f"   AUC-ROC: {result['auc']:.4f}\n")
                    f.write(f"   F1-Score: {result['f1']:.4f}\n")
                    
                    f.write(f"   Parameters:\n")
                    for param in ['dropout', 'modalities', 'filters', 'batch_size', 'patience', 'learning_rate', 'threshold']:
                        if param in result['parameters']:
                            f.write(f"     {param}: {result['parameters'][param]}\n")
        
        f.write(f"\n\nCompleted at: {datetime.now()}\n")
    
    print(f"📋 Final summary saved to: {summary_file}")

if __name__ == "__main__":
    create_sweep_script()
