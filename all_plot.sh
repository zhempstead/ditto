set -x

rm plots/*
rm corrs/*
python paper_plot.py

python crowd_independence.py ../gpt_di/entity_resolution/temp0-shots0
python crowd_independence.py ../gpt_di/entity_resolution/temp0-shots2

cp corrs/temp0-shots0.pdf plots/1.3c.pdf
cp corrs/temp0-shots2.pdf plots/1.3d.pdf
#cp corrs/crowd-temp0-shots2-results-wdc_unseen.png plots/1.1e.png
