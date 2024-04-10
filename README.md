# Open Category
## Chess Game Outcome Prediction Model
### Dataset
The dataset used for training and testing the model contains information about 15,000 chess games played. Features such as player ratings, time controls, opening moves, and game outcomes are included.

### Model Development
- **Data Preprocessing**: The dataset is preprocessed by handling missing values, encoding categorical variables, and scaling numerical features if necessary.
- **Feature Selection/Engineering**: Relevant features are selected for model training, including player ratings, time controls, and opening moves.
- **Model Selection**: Machine learning algorithms such as Random Forest, Decision Trees, SVM, and GBM are considered for classification.
- **Model Training**: The selected model is trained on the training data.
- **Model Evaluation**: The model's performance is evaluated using metrics such as accuracy, precision, recall, and F1-score.
<br><br>
# FE-SE Category
## Geometric Puzzle Solver Model
### Puzzle Representation
- The puzzle is represented using a suitable data structure that includes the constituent shapes, their positions, and orientations.
- Rules and constraints for rotating and rearranging each shape, as well as rules for their adjacency, are defined.

### Model Development
- **Data Reading**: Puzzle data is read from a CSV file containing information about the shapes, positions, and orientations.
- **Initialization**: The puzzle object is initialized with the puzzle state obtained from the CSV file.
- **Search Algorithms**: Search algorithms such as BFS, DFS, or A* are implemented to find the optimal solution with the fewest moves.
- **Model Evaluation**: The model's performance is evaluated on test cases to ensure efficient puzzle solving.

## Dependencies for Both Category
- Python 3.x
- Required libraries: pandas, scikit-learn
