for i in $(seq 1 8);
do
    python3 optuna_opt.py --exp-tag lidar-optuna&
done

