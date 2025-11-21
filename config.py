class Config:
    CLEANED_DATA_PATH = "data/cleaned_data.csv"
    DATA_PATH = "data/2*_en.csv"
    TEST_SIZE = 0.5
    N_ESTIMATORS = 100
    RANDOM_SEED = 42
    TENSORFLOW_EPOCHS = 500
    PYTORCH_EPOCHS = 100
    rename_columns={
        "MYR": "Model Year",
        "OL": "Overall Length",
        "OW": "Overall Width",
        "OH": "Overall Height",
        "WB": "Wheelbase",
        "CW": "Curb Weight",
        "A": "Front End Length",
        "B": "Rear End Length",
        "C": "Side Glass Height",
        "D": "Body Side Height",
        "E": "Roof Width",
        "F": "Front Overhang",
        "G": "Rear Overhang",
        "TWF": "Track Width Front",
        "TWR": "Track Width Rear",
        "WDIST": "Weight Distribution"
    }
    features = [
    'Overall Length', 'Overall Width', 'Overall Height',
    'Front End Length', 'Rear End Length',
    'Side Glass Height', 'Body Side Height',
    'Wheelbase', 'Front Overhang', 'Rear Overhang', # Sum = Overall Length
    'Roof Width', 'Track Width Front', 'Track Width Rear', # ~ Overall Width
    ]
    target = 'Curb Weight'