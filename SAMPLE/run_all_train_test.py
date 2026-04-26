from train_models import train_and_evaluate_all
import warnings
warnings.filterwarnings('ignore')

# Run with small config for quick test
results = train_and_evaluate_all(
    models=['lstm', 'gru', 'randomforest'],
    epochs=2,
    batch_size=64,
    lr=0.001,
    seq_len=12
)
print("\nFinal results:", results)
