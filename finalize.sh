#!/bin/bash
# Uso: ./finalize.sh  -> tabla/gráficas, notebook ejecutado, PDF, git y paquete
set -e
cd /home/claude/lab9
export OMP_NUM_THREADS=2
python run_06_report.py > logs_report.txt 2>&1
python build_notebook.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=900 Lab9_Series_de_Tiempo.ipynb
python reporte/build_report.py
