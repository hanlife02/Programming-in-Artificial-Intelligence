import pandas as pd
import matplotlib.pyplot as plt
import os

def load_loss_data(momentum):
    """Load loss data for a specific momentum value."""
    file_path = f'results/loss_{momentum}.csv'
    df = pd.read_csv(file_path)
    return df['Step'].values, df['Value'].values

def plot_all_momentum():
    """Plot loss curves for all momentum values in one figure."""
    plt.figure(figsize=(10, 6))
    
    momentums = [0.1, 0.5, 0.9]
    colors = ['blue', 'red', 'green']
    
    for momentum, color in zip(momentums, colors):
        steps, losses = load_loss_data(momentum)
        plt.plot(steps, losses, color=color, label=f'Momentum {momentum}', linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves for Different Momentum Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('loss_comparison_all_momentum.png', dpi=300, bbox_inches='tight')
    print("Generated: loss_comparison_all_momentum.png")
    plt.close()

def plot_momentum_09():
    """Plot loss curve for momentum 0.9 only."""
    plt.figure(figsize=(10, 6))
    
    steps, losses = load_loss_data(0.9)
    plt.plot(steps, losses, color='green', label='Momentum 0.9', linewidth=2)
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve (Momentum 0.9)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('loss_momentum_0.9.png', dpi=300, bbox_inches='tight')
    print("Generated: loss_momentum_0.9.png")
    plt.close()

if __name__ == "__main__":
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate both plots
    plot_all_momentum()
    plot_momentum_09()
    
    print("All plots generated successfully!")