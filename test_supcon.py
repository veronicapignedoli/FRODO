#!/usr/bin/env python3
"""
Quick integration test for SupCon loss and feature extraction.
Tests: forward pass with and without SupCon, shape correctness, loss computation.
"""
import torch
import torch.nn as nn
import sys

# Test imports
try:
    from architecture import MultiModalClassifier
    from losses import supervised_contrastive_loss
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Create model
model = MultiModalClassifier(in_channels_list=[1, 1], base_filters=16, num_classes=1, dropout=0.3).to(device)
print("✓ Model instantiated")

# Create dummy batch
B, D, H, W = 4, 32, 32, 32
flair = torch.randn(B, 1, D, H, W).to(device)
qsm = torch.randn(B, 1, D, H, W).to(device)
labels = torch.tensor([0, 1, 0, 1], dtype=torch.long).to(device)
print(f"✓ Dummy batch created: B={B}, D={D}, H={H}, W={W}")

# Test 1: Forward pass without features (default behavior)
print("\n--- Test 1: Forward pass without return_features ---")
try:
    logits = model([flair, qsm])
    assert logits.shape == (B, 1), f"Expected shape (B, 1), got {logits.shape}"
    print(f"✓ Logits shape: {logits.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 2: Forward pass with features
print("\n--- Test 2: Forward pass with return_features=True ---")
try:
    logits, z_fused = model([flair, qsm], return_features=True)
    assert logits.shape == (B, 1), f"Expected logits shape (B, 1), got {logits.shape}"
    assert z_fused.shape == (B, 512), f"Expected features shape (B, 512), got {z_fused.shape}"
    print(f"✓ Logits shape: {logits.shape}, Features shape: {z_fused.shape}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 3: SupCon loss computation
print("\n--- Test 3: SupCon loss computation ---")
try:
    loss = supervised_contrastive_loss(z_fused, labels, temperature=0.1)
    assert loss.shape == torch.Size([]), f"Expected scalar loss, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"✓ SupCon loss: {loss.item():.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 4: SupCon with single class (edge case)
print("\n--- Test 4: SupCon with single class (edge case) ---")
try:
    labels_single = torch.tensor([0, 0, 0, 0], dtype=torch.long).to(device)
    loss = supervised_contrastive_loss(z_fused, labels_single, temperature=0.1)
    assert loss.item() == 0.0, f"Expected 0 loss for single class, got {loss.item()}"
    print(f"✓ SupCon loss for single class: {loss.item():.4f} (correct: 0)")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 5: Learnable weight initialization
print("\n--- Test 5: Learnable weight initialization ---")
try:
    supcon_logit = torch.tensor(-2.5, dtype=torch.float32, device=device, requires_grad=True)
    w_supcon = torch.nn.functional.softplus(supcon_logit)
    assert w_supcon.item() > 0, "Weight should be positive"
    assert 0.05 <= w_supcon.item() <= 0.2, f"Weight should be small, got {w_supcon.item()}"
    print(f"✓ SupCon weight from logit -2.5: {w_supcon.item():.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

# Test 6: Total loss composition
print("\n--- Test 6: Total loss composition ---")
try:
    criterion = nn.BCEWithLogitsLoss()
    logits, z_fused = model([flair, qsm], return_features=True)
    labels_float = labels.float()
    
    loss_cls = criterion(logits.squeeze(1), labels_float)
    loss_supcon = supervised_contrastive_loss(z_fused, labels, temperature=0.1)
    w_supcon = torch.nn.functional.softplus(supcon_logit)
    loss_total = loss_cls + w_supcon * loss_supcon
    
    assert not torch.isnan(loss_total), "Total loss is NaN"
    assert loss_total.item() >= 0, "Total loss should be non-negative"
    print(f"✓ Loss composition:")
    print(f"  - Classification loss: {loss_cls.item():.4f}")
    print(f"  - SupCon loss: {loss_supcon.item():.4f}")
    print(f"  - SupCon weight: {w_supcon.item():.4f}")
    print(f"  - Total loss: {loss_total.item():.4f}")
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("✓ ALL TESTS PASSED")
print("="*60)
print("\nAll SupCon components are working correctly!")
