#!/usr/bin/env bash

# Requires imagemagick

f1=Results/Images/Weakly_Sup_CE_Loss
f2=Results/Images/Weakly_Sup_CE_Loss_SizePenalty

echo $f1 $f2

# convert Results/Images/Weakly_Sup_CE_Loss/0_Ep_*.png Results/Images/CE_loss_0.gif
# convert Results/Images/Weakly_Sup_CE_Loss_SizePenalty/0_Ep_*.png Results/Images/CE_loss_size_0.gif

rm -rf tmp/ && mkdir -p tmp/

for i in $f1/0_Ep_*.png ; do
    echo $i
    epc=`basename $i | cut -d . -f 1 | cut -d _ -f 3`
    convert $i -size 10x xc:none ${i/$f1/$f2} +append tmp/$epc.png
    mogrify -crop 530x518+0+0\! tmp/$epc.png
    mogrify -gravity north -extent 530x550 tmp/$epc.png
    mogrify -gravity south -extent 530x580 tmp/$epc.png
    mogrify -annotate +230+10 "Epoch $epc" tmp/$epc.png
    mogrify -annotate +100+560 "Partial CE" tmp/$epc.png
    mogrify -annotate +330+560 "Partial CE + Sizeloss" tmp/$epc.png
done

convert -loop 0 -delay 20 tmp/*.png result.gif

# rm -r tmp/