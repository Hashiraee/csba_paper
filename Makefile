SHELL=/bin/bash
VIRTUALENV?=env
.PHONY: help install clean envclean run freeze

help:
	@echo "Make targets:"
	@echo " install     	Create virtual environment (venv) and install required packages"
	@echo " freeze      	Persist installed packages to requirements.txt"
	@echo " clean       	Remove *.pyc files and __pycache__ directory"
	@echo " envclean 	Remove virtual envirnment (env)"
	@echo " run         	Run virtual environment"
	@echo "Check the Makefile for more details..."

install:
	@python3 -m venv $(VIRTUALENV)
	@. $(VIRTUALENV)/bin/activate; pip3 install --upgrade pip; pip3 install -r requirements.txt

freeze:
	@. $(VIRTUALENV)/bin/activate; pip3 freeze > requirements.txt

clean:
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} \+

envclean: clean
	@rm -rf $(VIRTUALENV)

run:
	@. $(VIRTUALENV)/bin/activate; python3 code/implementation.py
