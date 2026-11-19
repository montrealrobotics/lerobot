# Split checkpoints: solved (final>=0.8) vs failed (final<=0.2) seeds

## lamp arm1 @ 4000

| seed | final_train | first_success_step | never_succeeded | warmup_success | ma10_at_50k | ma10_at_100k | ma10_at_200k | alpha_final | alpha_early | alpha_min | entropy_final | entropy_early | critic_loss_final | critic_loss_early | actor_grad_final | actor_grad_early | critic_grad_final | critic_grad_early | ep_length_final | ep_length_early | ep_return_final | ep_return_early |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1 | 2.75e+03 | 0 | 1 | 0 | 0 | 0.1 | 0.00746 | 0.668 | 0.00723 | -15.9 | 21.3 | 0.167 | 16.2 | 2.39 | 20.2 | 1.25 | 56.2 | 59.2 | 970 | 0 | 0 |
| 1 | 0 | 2.36e+03 | 0 | 1 | 0 | 0.1 | 0.1 | 0.00783 | 0.676 | 0.00755 | -16.1 | 21.4 | 1.17 | 22.2 | 6.34 | 16.9 | 6.18 | 77.8 | 973 | 923 | 0 | 0 |
| 2 | 1 | 579 | 0 | 1 | 0.1 | 0.1 | 0.2 | 0.00775 | 0.669 | 0.00758 | -16 | 21.3 | 0.141 | 16.6 | 2.42 | 22 | 1.66 | 48 | 64 | 760 | 0 | 0 |

## lamp arm3 @ 4000

| seed | final_train | first_success_step | never_succeeded | warmup_success | ma10_at_50k | ma10_at_100k | ma10_at_200k | alpha_final | alpha_early | alpha_min | entropy_final | entropy_early | critic_loss_final | critic_loss_early | actor_grad_final | actor_grad_early | critic_grad_final | critic_grad_early | ep_length_final | ep_length_early | ep_return_final | ep_return_early |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 302 | 0 | 1 | 0 | 0 | 0 | 0.00739 | 0.676 | 0.00712 | -16.1 | 21.5 | 0.561 | 13 | 5.35 | 17.5 | 4.25 | 51.9 | 983 | 836 | 0 | 0 |
| 1 | 0 | 4.69e+04 | 0 | 0 | 0.1 | 0 | 0 | 0.00749 | 0.676 | 0.00725 | -16.1 | 21.5 | 0.553 | 10.8 | 5.31 | 17.1 | 4.31 | 48.9 | 990 | 977 | 0 | 0 |
| 2 | 1 | 681 | 0 | 1 | 0.3 | 0 | 0 | 0.00757 | 0.676 | 0.00713 | -16.2 | 21.6 | 0.399 | 12.6 | 4.42 | 16.3 | 3.88 | 33.2 | 91.9 | 961 | 0 | 0 |

## lamp arm4 @ 2000

| seed | final_train | first_success_step | never_succeeded | warmup_success | ma10_at_50k | ma10_at_100k | ma10_at_200k | alpha_final | alpha_early | alpha_min | entropy_final | entropy_early | critic_loss_final | critic_loss_early | actor_grad_final | actor_grad_early | critic_grad_final | critic_grad_early | ep_length_final | ep_length_early | ep_return_final | ep_return_early |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.88 | 802 | 0 | 1 | 0.1 | 0 | 0 | 0.00778 | 0.676 | 0.00685 | -16.3 | 21.4 | 0.608 | 11.9 | 5.19 | 12.5 | 4.6 | 25.8 | 75.2 | 930 | 0 | 0 |
| 1 | 0 | 1.8e+03 | 0 | 1 | 0.1 | 0 | 0.3 | 0.00745 | 0.676 | 0.00696 | -16 | 21.4 | 0.572 | 12.6 | 3.93 | 12.3 | 3.27 | 32.9 | 990 | 946 | 0 | 0 |
| 2 | 1 | 714 | 0 | 1 | 0 | 0.1 | 0.7 | 0.00747 | 0.669 | 0.00732 | -15.9 | 21.3 | 0.175 | 16.6 | 2.19 | 22.4 | 1.5 | 48.8 | 54.6 | 589 | 0 | 0 |
