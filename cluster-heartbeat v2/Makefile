.PHONY: setup data train infer serve test docker clean

setup:        ## install the package (editable) + dev deps
	pip install -e . && pip install -r requirements.txt

data:         ## generate the synthetic DCGM dataset
	python scripts/generate_synthetic_data.py

train:        ## train the heartbeat model → checkpoints/best
	python scripts/train.py

pca:          ## train the Phase-1 PCA baseline
	python scripts/train.py --set model.type=pca

infer:        ## fleet inference → reports/latest_report.json
	python scripts/run_inference.py

serve:        ## serve the API on :8000
	python scripts/serve.py --port 8000

test:         ## run the test suite
	pytest -q

docker:       ## build the all-in-one image
	docker build -t cluster-heartbeat .

clean:
	rm -rf checkpoints logs reports data/synthetic/*.csv data/synthetic/failures.json
