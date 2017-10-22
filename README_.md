Step 1: make_dataframe -> create dataframe csv
Step 2: (use datafram csv) extract_cubes_by_label -> extract cubes to /resources/_tfrecords/
Step 3: remove_empyt_tfrecords -> removes empty tfrecords AND evens out files by label type
Step 4: train_label_cat -> trains with only malignacy and across 6 distict categories