#!/usr/bin/env bash

# Requires ImageMagick

f1=results/images/TOY/unconstrained
f2=results/images/TOY/constrained

echo $f1 $f2

rm -rf tmp/ && mkdir -p tmp/

for i in $f1/1_Ep_*.png ; do
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