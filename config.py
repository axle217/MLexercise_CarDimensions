class Config:
    DATA_PATH = "data/cleaned_data.csv"
    TEST_SIZE = 0.5
    N_ESTIMATORS = 100
    RANDOM_SEED = 42
    TENSORFLOW_EPOCHS = 500
    PYTORCH_EPOCHS = 100
    features = [
    'Overall Length', 'Overall Width', 'Overall Height',
    'Front End Length', 'Rear End Length',
    'Side Glass Height', 'Body Side Height',
    'Wheelbase', 'Front Overhang', 'Rear Overhang', # Sum = Overall Length
    'Roof Width', 'Track Width Front', 'Track Width Rear', # ~ Overall Width
    ]
    target = 'Curb Weight'