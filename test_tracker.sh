max_age=(1 5 20)
min_hits=(3 5 7)
iou_thres=0.2

benchmark_name="MOT15_Train"
dataset="data/MOT15/train/"

samples=`ls $dataset`
for age in ${max_age[@]}; do
    for min_hit in ${min_hits[@]}; do
        dir="${benchmark_name}_${age}_${min_hit}_$iou_thres/data"
        mkdir -p $dir
        for sample in $samples; do
            input="$dataset$sample/img1/"
            echo "-i $input -ev "$dir/$sample.txt" --max_age $age --min_hits $min_hit --iou_threshold $iou_thres"
            poetry run python pedestrian_tracker.py -i $input -ev "$dir/$sample.txt" --max_age $age --min_hits $min_hit --iou_threshold $iou_thres
        done
    done
done
