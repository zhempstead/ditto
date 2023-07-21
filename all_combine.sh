set -x

for exp in temp0-cot0 temp0-cot2 temp0-shots0 temp0-shots2 temp0-shots2c temp2-shots0 temp2-shots2 valid-temp0-shots2
do
    python crowd_gpt.py combine --rawdir ../raw_gpt_di/entity_resolution/${exp}-raw/ --outdir ../gpt_di/entity_resolution/${exp} &
done
wait

for i in {0..9}
do
    python crowd_gpt.py combine --rawdir ../raw_gpt_di/entity_resolution/temp0-shots2u${i}-raw --outdir ../gpt_di/entity_resolution/fixed_shots/${i}/ &
done
wait
