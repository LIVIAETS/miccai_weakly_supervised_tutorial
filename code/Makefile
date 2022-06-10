
data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	cp -r $@_tmp/train/gt $@_tmp/train/weak  # Not used but makes the code easier
	cp -r $@_tmp/val/gt $@_tmp/val/weak
	mv $@_tmp $@

# Extraction and slicing for PROMISE12
data/PROMISE12: data/promise12
	rm -rf $@_tmp
	python3 slice_promise.py --source_dir $< --dest_dir $@_tmp
	mv $@_tmp $@
data/promise12: data/promise12.lineage data/TrainingData_Part1.zip data/TrainingData_Part2.zip data/TrainingData_Part3.zip
	md5sum -c $<
	rm -rf $@_tmp
	unzip -q $(word 2, $^) -d $@_tmp
	unzip -q $(word 3, $^) -d $@_tmp
	unzip -q $(word 4, $^) -d $@_tmp
	mv $@_tmp $@

# Extraction and slicing for ACDC
data/ACDC: data/acdc
	$(info $(yellow)$(CC) $(CFLAGS) preprocess/slice_acdc.py$(reset))
	rm -rf $@_tmp $@
	python3 slice_acdc.py --source_dir="data/acdc/training" --dest_dir=$@_tmp \
		--seed=0 --retains 10 --retains_test 20
	mv $@_tmp $@
data/acdc: data/acdc.lineage data/acdc.zip
	$(info $(yellow)unzip data/acdc.zip$(reset))
	md5sum -c $<
	rm -rf $@_tmp $@
	unzip -q $(word 2, $^) -d $@_tmp
	rm $@_tmp/training/*/*_4d.nii.gz  # space optimization
	mv $@_tmp $@


results.gif: results/images/TOY/unconstrained results/images/TOY/constrained
	./gifs.sh

results/images/TOY/unconstrained: data/TOY
	python3 main.py --dataset TOY --mode unconstrained --gpu

results/images/TOY/constrained: data/TOY
	python3 main.py --dataset TOY --mode constrained --gpu