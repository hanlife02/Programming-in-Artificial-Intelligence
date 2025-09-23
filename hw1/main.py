def main():
    print("=== CIFAR-10 LeNet 训练和可视化 ===\n")
    
    # 1. 运行训练
    print("开始训练模型...")
    try:
        from train import train_model
        train_model()
        print("\n✓ 模型训练完成！")
    except ImportError as e:
        print(f"错误：无法导入训练模块 - {e}")
        return
    except Exception as e:
        print(f"训练过程中出现错误：{e}")
        return
    
    # 2. 绘制损失曲线
    print("\n开始绘制损失曲线...")
    try:
        from draw import plot_loss_curves
        plot_loss_curves()
        print("\n✓ 损失曲线绘制完成！")
    except ImportError as e:
        print(f"错误：无法导入绘图模块 - {e}")
    except Exception as e:
        print(f"绘图过程中出现错误：{e}")
    
    print("\n=== 所有任务完成 ===")

if __name__ == '__main__':
    main()