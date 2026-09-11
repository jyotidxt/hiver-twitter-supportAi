.PHONY: help install preprocess eda annotate train run evaluate judge report test validate clean

python ?= python

help:
	@echo "Hiver Support AI — Available Commands:"
	@echo "  make install     - Install Python dependencies"
	@echo "  make preprocess  - Run data engineering preprocessing pipeline"
	@echo "  make eda         - Run EDA analytics & report generator"
	@echo "  make annotate    - Launch terminal annotation tool"
	@echo "  make train       - Train intent classifier baseline"
	@echo "  make run         - Run interactive AI Agent CLI demo"
	@echo "  make evaluate    - Run quantitative evaluation harness & baselines"
	@echo "  make judge       - Run qualitative LLM-as-a-Judge & human agreement"
	@echo "  make report      - Assembled final REPORT.md document"
	@echo "  make test        - Run complete 93-test pytest suite"
	@echo "  make validate    - Verify repository structure and submission checklist"

install:
	$(python) -m pip install -r requirements.txt

preprocess:
	$(python) -m src.preprocessing.run_pipeline

eda:
	$(python) -m src.analysis.run_eda

annotate:
	$(python) -m src.annotation.annotation_tool

train:
	$(python) -c "from src.classifier.intent_classifier import IntentClassifier; clf = IntentClassifier(); clf.train(); print('Classifier trained.')"

run:
	$(python) -m src.cli.run_agent

evaluate:
	$(python) -c "from src.evaluation.evaluator import EvaluationHarness; from src.evaluation.report import generate_evaluation_summary; h = EvaluationHarness(); res = h.run_evaluations(); generate_evaluation_summary(res)"

judge:
	$(python) -c "from src.judge.report import generate_judge_report; generate_judge_report()"

report:
	$(python) -m src.report.builder

test:
	pytest tests/ -v

validate:
	$(python) -m src.utils.validate_submission

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
