import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# =====================
# SEED
# =====================
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

print(f"PyTorch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


# ==========================================================
# Simple MLP for PCA track features -> 5 track parameters
# ==========================================================
class MLP(nn.Module):
    """
    Fully connected regression network

    input_dim  = number of PCA features
    output_dim = 5 track parameters
    width      = neurons per hidden layer
    depth      = number of hidden layers
    """

    def __init__(
        self,
        input_dim,
        output_dim=5,
        width=256,
        depth=4,
        activation="relu",
        use_batchnorm=False,
        dropout=0.0
    ):
        super().__init__()

        # choose activation
        if activation.lower() == "relu":
            act_layer = nn.ReLU
        elif activation.lower() == "tanh":
            act_layer = nn.Tanh
        elif activation.lower() == "gelu":
            act_layer = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        layers = []

        # first hidden layer
        layers.append(nn.Linear(input_dim, width))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(act_layer())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # remaining hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # output layer: predicts all 5 params at once
        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ==========================================================
# LOSS FUNCTION
# ==========================================================
def loss_fn(pred, target):
    """
    Basic supervised regression loss.
    This is just mean squared error over all 5 outputs.
    """
    return torch.mean((pred - target) ** 2)


# ==========================================================
# EXAMPLE CONSTANTS
# ==========================================================
INPUT_DIM = 32          # number of PCA features
OUTPUT_DIM = 5          # 5 track parameters
WIDTH = 128
DEPTH = 6
ACTIVATION = "relu"
USE_BATCHNORM = True
DROPOUT = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5     # L2 regularization
BATCH_SIZE = 256
EPOCHS = 50


# ==========================================================
# MODEL / OPTIMIZER
# ==========================================================
model = MLP(
    input_dim=INPUT_DIM,
    output_dim=OUTPUT_DIM,
    width=WIDTH,
    depth=DEPTH,
    activation=ACTIVATION,
    use_batchnorm=USE_BATCHNORM,
    dropout=DROPOUT
).to(device)

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY   # L2 regularization
)


# ==========================================================
# DUMMY EXAMPLE DATA
# replace this with your real PCA features and targets
# X shape = (N, INPUT_DIM)
# y shape = (N, 5)
# ==========================================================
N = 5000
X = torch.randn(N, INPUT_DIM)
y = torch.randn(N, OUTPUT_DIM)

dataset = torch.utils.data.TensorDataset(X, y)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ==========================================================
# TRAIN LOOP
# ==========================================================
def train(model, loader, optimizer, epochs=EPOCHS):
    model.train()

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_seen = 0

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)
            loss = loss_fn(pred, yb)

            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            n_seen += bs

        epoch_loss = running_loss / n_seen

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Loss = {epoch_loss:.6e}")


# ==========================================================
# TEST FUNCTION
# ==========================================================
def test_model(model, X_test, y_test):
    model.eval()

    with torch.no_grad():
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        pred = model(X_test)

        mse = torch.mean((pred - y_test) ** 2).item()
        rmse_per_param = torch.sqrt(torch.mean((pred - y_test) ** 2, dim=0))

    print("\n===== Test Report =====")
    print(f"Total MSE = {mse:.6e}")
    print(f"RMSE per parameter = {rmse_per_param.cpu().numpy()}")


# ==========================================================
# RUN
# ==========================================================
train(model, loader, optimizer, epochs=EPOCHS)

X_test = torch.randn(1000, INPUT_DIM)
y_test = torch.randn(1000, OUTPUT_DIM)
test_model(model, X_test, y_test)