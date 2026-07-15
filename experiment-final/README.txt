Replace these files in the project root:

- pipeline.py
- kg_modifier.py
- analyze_data.py
- analyze_with_learning.py
- compare_methods.py
- view_results.py

Full run:

python pipeline.py \
  --basepath ./100people_window_comparison \
  --experiment-name 100people_window_comparison \
  --n-people 100 \
  --n-vertices 100 \
  --n-variations 5 \
  --removal-percent-step 15 \
  --relation hasAge \
  --embedding-model mure \
  --embedding-epochs 100 \
  --embedding-dimensions 300 \
  --regression-model ridge \
  --max-k 10 \
  --seed 42 \
  --processes 8

Use --force only when regenerating from the KG step.

Resume examples:

python pipeline.py --basepath ./100people_window_comparison --start-step embeddings
python pipeline.py --basepath ./100people_window_comparison --start-step query-points
python pipeline.py --basepath ./100people_window_comparison --start-step analysis
python pipeline.py --basepath ./100people_window_comparison --start-step comparisons
python pipeline.py --basepath ./100people_window_comparison --start-step report
