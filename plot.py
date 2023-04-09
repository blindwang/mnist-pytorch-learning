from utils import plot_loss_acc

# 绘制loss和acc曲线
# https://tensorboard.dev/experiment/jwCBnTY2RhKD9XOvTq2flg/#scalars
experiment_id = "jwCBnTY2RhKD9XOvTq2flg"
plot_loss_acc(experiment_id, "compare_optim.csv", "compare_optim.png")