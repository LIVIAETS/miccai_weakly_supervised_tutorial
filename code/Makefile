
data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/PROMISE12:

# Extraction and slicing
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

results.gif: results/images/TOY/unconstrained results/images/TOY/constrained
	./gifs.sh

results/images/TOY/unconstrained: data/TOY
	python3 main.py --dataset TOY --mode unconstrained --gpu

results/images/TOY/constrained: data/TOY
	python3 main.py --dataset TOY --mode constrained --gpu