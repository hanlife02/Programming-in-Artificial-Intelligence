import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss_curves(csv_file='results/training_data.csv'):
    """
    从CSV文件读取训练数据并绘制损失曲线
    """
    # 检查CSV文件是否存在
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run the training script first.")
        return
    
    # 读取CSV数据
    try:
        data = pd.read_csv(csv_file)
        print(f"Successfully loaded data from {csv_file}")
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # 确保results目录存在
    os.makedirs('results', exist_ok=True)
    
    # 创建图表
    plt.figure(figsize=(8, 6))
    
    # 绘制训练和验证损失
    plt.plot(data['epoch'], data['train_loss'], 'b-', label='Training Loss', marker='o')
    plt.plot(data['epoch'], data['val_loss'], 'r-', label='Validation Loss', marker='s')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    
    # 保存为PNG格式
    png_path = 'results/loss_curve.png'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    print(f"Loss curve saved to: {png_path}")
    
    # 显示图表
    plt.show()
    
    # 打印训练总结
    print("\nTraining Summary:")
    print(f"Final Training Loss: {data['train_loss'].iloc[-1]:.4f}")
    print(f"Final Validation Loss: {data['val_loss'].iloc[-1]:.4f}")
    print(f"Lowest Training Loss: {data['train_loss'].min():.4f} (Epoch {data.loc[data['train_loss'].idxmin(), 'epoch']})")
    print(f"Lowest Validation Loss: {data['val_loss'].min():.4f} (Epoch {data.loc[data['val_loss'].idxmin(), 'epoch']})")

if __name__ == '__main__':
    plot_loss_curves()