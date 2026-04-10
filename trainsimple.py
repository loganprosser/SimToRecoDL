import os
import torch
import torch.optim as optim
import torch.nn as nn

from model import SimpleTrackNet
from helpers import denormalize_targets, save_model_checkpoint, wrapped_angle_diff
from helpers_data import load_track_data, print_data_shapes, set_seed
from helpers_vis import make_val_diagnostic_plots, print_final_validation_samples

# ===== Constants ======
EPOCHS = 100
HIDDEN_LAYERS = [512, 512, 128]
USE_BATCHNORM = False
DROPOUT = 0.0
SAVE_DIR = "modelsimple"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = os.path.join(SAVE_DIR, "simple_tracknet.pt")

# ====== Running flags ======
PRINT_FINAL_VAL_SAMPLES = False # not working need sigma for the funciton
SAVE_MODEL = True


# ===== Picking Device ========
device = torch.device(
    "mps" if torch.backends.mps.is_available() 
    else "cuda" if torch.cuda.is_available() 
    else "cpu"
    
)
print(F"Device set to {device}")

# ==== Setting Seed =====
SEED = 42

set_seed(SEED)


# ====== Load and prepare data =======
BATCH_SIZE = 256 #not sure why this effects model so much idk??
data = load_track_data(batch_size=BATCH_SIZE, seed=SEED, device=device)

train_loader = data.train_loader
val_loader = data.val_loader
X_train = data.x_train
X_val = data.x_val
Y_train = data.y_train
Y_val = data.y_val
x_mean = data.x_mean
x_std = data.x_std
y_mean = data.y_mean
y_std = data.y_std
y_mean_t = data.y_mean_t
y_std_t = data.y_std_t
FEATURE_COLS = data.feature_cols
TARGET_COLS = data.target_cols
PHI_INDEX = data.phi_index


# ====== CHECK SHAPES ======
CHECK_SHAPE = False
if CHECK_SHAPE:
    print_data_shapes(data)

# ===== Training ======

input_dim = X_train.shape[1]
# === Init Model =====

#model = TestTrackNet(input_dim=input_dim, hidden_dim=64, output_dim=5)
model = SimpleTrackNet(
    input_dim=input_dim,
    hidden_layers=HIDDEN_LAYERS,   #128, 128, 64]. [256, 256, 64]
    use_batchnorm=USE_BATCHNORM,
    dropout=DROPOUT,
    activation=nn.ReLU
)


model.to(device)

print(model)


# ====== Set up the Loss and optimizer =======

criterion = nn.MSELoss() # use for simple models
#optimizer = optim.Adam(model.parameters(), lr=1e-3) # use for simple models

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)


# ====== trial forward pass ======
TEST_TRAIN = False
if TEST_TRAIN:
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)
    preds = model(xb)

    print("pred shape:", preds.shape)
    print("target shape:", yb.shape)

    loss = criterion(preds, yb)
    print("initial loss:", loss.item())

# ===== Training loop =====
EPOCHS = EPOCHS

for epoch in range(EPOCHS):
    # ===== TRAIN ======
    model.train()
    train_loss = 0.0
    
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * xb.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    # ==== VALIDATION ====
    
    model.eval()
    val_loss = 0.0

    total_val_mae = torch.zeros(len(TARGET_COLS), device=device)
    total_val_sq  = torch.zeros(len(TARGET_COLS), device=device)
    total_count = 0

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss += loss.item() * xb.size(0)

            # ==== DENORMALIZE (if using normalization) ====
            preds_phys = denormalize_targets(preds, y_mean_t, y_std_t)
            yb_phys    = denormalize_targets(yb, y_mean_t, y_std_t)

            diff = preds_phys - yb_phys

            # wrap phi correctly
            diff[:, PHI_INDEX] = wrapped_angle_diff(
                preds_phys[:, PHI_INDEX],
                yb_phys[:, PHI_INDEX]
            )

            total_val_mae += diff.abs().sum(dim=0)
            total_val_sq  += (diff ** 2).sum(dim=0)
            total_count   += xb.size(0)

    val_loss /= len(val_loader.dataset)

    per_target_mae = (total_val_mae / total_count).detach().cpu().numpy()
    per_target_rmse = torch.sqrt(total_val_sq / total_count).detach().cpu().numpy()

    overall_val_mae = per_target_mae.mean()
    overall_val_rmse = per_target_rmse.mean()
    
    print(f"EPOCH {epoch + 1:2d}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Mean MAE: {overall_val_mae:.6f} | Val Mean RMSE: {overall_val_rmse:.6f}")

    print("   Per-target MAE:")
    for name, val in zip(TARGET_COLS, per_target_mae):
        print(f"      {name}: {val:.6f}")

    print("   Per-target RMSE:")
    for name, val in zip(TARGET_COLS, per_target_rmse):
        print(f"      {name}: {val:.6f}")
    
if PRINT_FINAL_VAL_SAMPLES:
    print_final_validation_samples(
        model, val_loader, device,
        y_mean_t, y_std_t,
        TARGET_COLS, PHI_INDEX,
        num_examples=4
    )
    
if SAVE_MODEL:
    save_model_checkpoint(
        save_path=MODEL_PATH,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=EPOCHS,
        metadata={
            "model_type": "SimpleTrackNet",
            "input_dim": input_dim,
            "output_dim": len(TARGET_COLS),
            "hidden_layers": HIDDEN_LAYERS,
            "feature_cols": FEATURE_COLS,
            "target_cols": TARGET_COLS,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "use_batchnorm": USE_BATCHNORM,
            "dropout": DROPOUT,
            "activation": "ReLU",
            "batch_size": BATCH_SIZE,
            "seed": SEED,
            "val_fraction": 0.2,
            "criterion": criterion.__class__.__name__,
        },
    )

    print(f"Model saved to {MODEL_PATH}")
    
# ====== SAVE VALIDATION DISTRIBUTION PLOTS ======
SAVE_PLOTS = True

if SAVE_PLOTS:
    plot_paths = make_val_diagnostic_plots(
        model=model,
        val_loader=val_loader,
        device=device,
        y_mean_t=y_mean_t,
        y_std_t=y_std_t,
        target_cols=TARGET_COLS,
        phi_index=PHI_INDEX,
        output_dir="plots",
        prefix="simple_val",
        bins=100,
        density=True,
        show=False  # set True if you want popup
    )

    print("Saved validation diagnostic plots:")
    for plot_name, plot_path in plot_paths.items():
        print(f"  {plot_name}: {plot_path}")
