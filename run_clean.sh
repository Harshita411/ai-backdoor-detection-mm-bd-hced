echo ----------------------------start training ------------------------------

for((integer = 0; integer <= 2; integer ++))
do

  foo2="python train.py --model_dir clean$integer"
  $foo2
done
