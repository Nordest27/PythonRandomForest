# PythonRandomForest
Random forests implemented from scratch in Python

## Files and folders

The project is organized in the following folders and files:

 - `src/`: Folder containing the Random Forest implementation.
 - `src/class_reg_decision_tree.py`: Decision Tree class.
 - `src/random_forest.py`: Random Forest class.
 - `src/main.py`: Main Python scrip that runs the Random Forest with a given dataset.
 - `test/`: Folder containing some unit tests.
 - `test/covertype_test.py`: Unit test using the Covertype dataset
 - `test/income_test.py`: Unit test using the Adult dataset
 - `test/iris_test.py`: Unit test using the Iris dataset
 - `test/letter_classify.py`: Unit test using the Letter dataset
 - `test/students_test.py`: Unit test using the Students (ucimlrepo) dataset
 - `test/titanic_test.py`: Unit test using the Titanic dataset
 - `requirements.txt`: File containing all the project's requirements. This file can be pip installable.

## How to run the code

To run the code we suggest creating a new Python environment and still the requirements by running the following commands from the top-level folder:

```
python3 -m venv [env-name]
source env-name/bin/activate
pip install -r requirements.txt
```

Once we have the requirements installed, we can execute the code by running `python main.py`.

By default, the code uses the Letter dataset, but this can be changed by modifying the 7th line of the `src/main.py` script.
